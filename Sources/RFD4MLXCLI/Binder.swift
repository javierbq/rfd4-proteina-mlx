import Foundation
import MLX
import RFD4Core
import RFD4MLX

/// Conditioned binder design: a target structure, a contig, and hotspots.
///
/// The shipped recipe is `"50-120, /0, A17-131"` against PD-L1 with three hotspot
/// sidechains. Everything the model needs beyond the layout comes from the target:
/// its backbone geometry as one rigid group with a free pose, its sequence, and the
/// atoms the design should engage.
enum BinderCommand {

    struct ResidueKey: Hashable { let chain: String; let residue: Int }

    struct Options {
        var weights: String
        var target: String
        var out: String
        var contig = "50-120, /0, A17-131"
        /// `chain/residue/atom` triples, as the recipe's selections name them.
        var hotspots: [(chain: String, residue: Int, atom: String)] = [
            ("A", 56, "CG"), ("A", 56, "OH"),
            ("A", 115, "CG"), ("A", 115, "SD"),
            ("A", 123, "CD2"), ("A", 123, "OH"),
        ]
        /// Which atoms of the target carry its internal geometry.
        var backboneAtoms: Set<String> = ["N", "CA", "C", "O"]
        var designLength: Int?
        var seed: UInt64 = 0
        var strategy: ReleasedRecycleStrategy = .thinkingLow
        /// How many designs to produce. Seeds run `seed ..< seed + count`.
        var count = 1
        /// Where per-design metrics are appended, one row each.
        var statsPath: String?

        /// `A245:CB,SG A248:CA` — the form the batch runner takes on the command line.
        static func parseHotspots(_ spec: String)
            -> [(chain: String, residue: Int, atom: String)] {
            var out: [(chain: String, residue: Int, atom: String)] = []
            for group in spec.split(separator: " ") {
                let parts = group.split(separator: ":")
                guard parts.count == 2 else { continue }
                let head = String(parts[0])
                let chain = String(head.prefix(while: { $0.isLetter }))
                guard let residue = Int(head.dropFirst(chain.count)) else { continue }
                for atom in parts[1].split(separator: ",") {
                    out.append((chain, residue, String(atom)))
                }
            }
            return out
        }
    }


    struct Built {
        let problem: DesignProblem
        let hotspots: [(token: Int, slot: Int)]
    }

    /// Hang the target's atoms on the tokens that preserve them.
    ///
    /// Called once per design because the contig redraws the binder length each
    /// seed, so the token indices shift and every array here is layout-shaped.
    static func build(
        layout: TokenLayout,
        atomsByResidue: [ResidueKey: [(slot: Int, name: String, coord: Vec3)]],
        residueName: [ResidueKey: String],
        options o: Options
    ) throws -> Built {
        var coords = [[Vec3?]](repeating: [Vec3?](repeating: nil, count: Vocabulary.slotCount),
                               count: layout.count)
        var backbone: [(token: Int, slot: Int)] = []
        var hotspots: [(token: Int, slot: Int)] = []
        var knownSequence = [Int?](repeating: nil, count: layout.count)
        let wanted = Set(o.hotspots.map { "\($0.chain)/\($0.residue)/\($0.atom)" })
        var unresolved = 0

        // A preserved token's run takes its id from the preserved segment itself, so
        // `token.chainID` is the INPUT chain for exactly the tokens that have an
        // input residue — which is every token this loop acts on.
        for (t, token) in layout.tokens.enumerated() {
            guard let rid = token.inputResidue else { continue }
            let key = ResidueKey(chain: token.chainID, residue: rid)
            guard let atoms = atomsByResidue[key] else { continue }
            for atom in atoms {
                coords[t][atom.slot] = atom.coord
                // `not has_nan_coord()` in the shipped recipe's selection: an
                // unresolved atom makes the whole group's centroid NaN.
                let resolved = atom.coord.x.isFinite && atom.coord.y.isFinite
                    && atom.coord.z.isFinite
                if o.backboneAtoms.contains(atom.name) {
                    if resolved { backbone.append((t, atom.slot)) } else { unresolved += 1 }
                }
                if wanted.contains("\(token.chainID)/\(rid)/\(atom.name)") {
                    hotspots.append((t, atom.slot))
                }
            }
            if let name = residueName[key],
               let channel = DesignProblem.sequenceProbabilityChannels.firstIndex(of: name) {
                knownSequence[t] = channel
            }
        }
        guard !backbone.isEmpty else {
            throw CLIError.message("no target backbone atoms matched the contig's residues")
        }
        // DISTINCT pairs, not matches: a duplicate row must not make up for an atom
        // that is genuinely absent.
        struct Flagged: Hashable { let token: Int; let slot: Int }
        let distinct = Set(hotspots.map { Flagged(token: $0.token, slot: $0.slot) })
        guard distinct.count == o.hotspots.count else {
            throw CLIError.message("""
                flagged \(distinct.count) distinct hotspot atoms of the \
                \(o.hotspots.count) requested — check the residue numbering of \(o.target)
                """)
        }
        return Built(
            problem: DesignProblem(
                layout: layout, coords: coords,
                internalGroups: [DesignProblem.CoordinateGroup(members: backbone)],
                hotspots: hotspots, knownSequence: knownSequence),
            hotspots: hotspots)
    }

    static func run(_ o: Options) throws {
        let vocab = Vocabulary.unifiedAtom37
        let raw = try AtomArray.fromMMCIF(contentsOf: URL(fileURLWithPath: o.target))
        // The two parse-time transforms upstream's design loader applies. Without
        // them a selenomethionine residue vanishes from the chain and an alternate
        // conformer is counted as a second atom; see `normalizedForDesign`.
        let target = raw.normalizedForDesign()
        print("target: \(target.count) atoms from \(o.target)"
              + (target.count != raw.count
                 ? "  (\(raw.count - target.count) alternate-conformer rows collapsed)" : ""))

        // Atoms indexed by (chain, residue). The chain belongs in the key: upstream
        // selects `chain_id == c & start <= res_id <= end`, and a multi-chain input
        // routinely numbers two chains over the same range, so dropping the chain
        // merges residues from a chain the contig never asked for.
        var residueName: [ResidueKey: String] = [:]
        var atomsByResidue: [ResidueKey: [(slot: Int, name: String, coord: Vec3)]] = [:]
        var residuesByChain: [String: Set<Int>] = [:]
        let slots = target.atom37Slots(vocab)
        for i in target.indices {
            guard let rid = target.resID?[i], rid >= 0 else { continue }
            let slot = slots[i]
            guard slot >= 0 else { continue }
            let chain = target.chainID?[i] ?? ""
            let key = ResidueKey(chain: chain, residue: rid)
            if residueName[key] == nil {
                residueName[key] = target.resName[i]
                residuesByChain[chain, default: []].insert(rid)
            }
            atomsByResidue[key, default: []]
                .append((slot, target.atomName[i].trimmingCharacters(in: .whitespaces),
                         target.coord[i]))
        }
        for chain in residuesByChain.keys.sorted() {
            let present = residuesByChain[chain]!.sorted()
            let span = (present.last ?? 0) - (present.first ?? 0) + 1
            let gaps = span - present.count
            print("target chain \(chain): \(present.count) residues "
                  + "(\(present.first ?? -1)…\(present.last ?? -1))"
                  + (gaps > 0 ? "  — \(gaps) unmodelled" : ""))
        }

        // Weights once, not once per design: loading is mmap'd and cheap, but the
        // first forward pass compiles Metal kernels, and paying that 100 times is a
        // real cost for no reason.
        let loadStart = Date()
        let generator = try Generator(weightsAt: URL(fileURLWithPath:
            (o.weights as NSString).expandingTildeInPath))
        print("loaded weights in \(String(format: "%.1f", Date().timeIntervalSince(loadStart)))s")

        if let statsPath = o.statsPath, !FileManager.default.fileExists(atPath: statsPath) {
            let header = "seed,binder_len,tokens,seconds,caca_mean,caca_min,caca_max,"
                + "binder_rg,iface_min,contacts_10a,hotspot_med,other_med,peak_gb\n"
            try header.write(toFile: statsPath, atomically: true, encoding: .utf8)
        }

        let batchStart = Date()
        for i in 0 ..< o.count {
            let seed = o.seed &+ UInt64(i)
            // The contig's 60-80 is redrawn per seed, so each design is its own
            // length unless the caller pinned one.
            var rng = SplitMix64(seed: seed)
            let layout = try TokenLayout.make(
                contig: o.contig,
                residuesInChain: { residuesByChain[$0].map(Array.init) ?? [] },
                draw: { lo, hi in
                    o.designLength ?? (lo == hi ? lo : lo + Int(rng.next() % UInt64(hi - lo + 1)))
                })
            let built = try build(layout: layout, atomsByResidue: atomsByResidue,
                                  residueName: residueName, options: o)
            let system = try built.problem.validatedSystem(seed: seed)

            let designed = layout.designedMask().filter { $0 }.count
            let runStart = Date()
            let result = try generator.generate(system, strategy: o.strategy, seed: seed) { _, _ in }
            let elapsed = Date().timeIntervalSince(runStart)

            let out = o.count == 1 ? o.out
                : o.out.replacingOccurrences(of: ".cif", with: String(format: "_%03d.cif", i))
            try writeComplex(result, layout: layout, vocab: vocab, to: out)

            let m = metrics(result, chains: layout.chainRanges(),
                            hotspotTokens: Set(built.hotspots.map(\.token)))
            print(String(format: "  [%3d/%d] seed %llu  %d aa  %d tok  %6.1f s  "
                         + "CA-CA %.2f  Rg %.1f  iface %.2f  contacts %d  hot %.1f/%.1f",
                         i + 1, o.count, seed, designed, layout.count, elapsed,
                         m.caca, m.rg, m.ifaceMin, m.contacts, m.hotMed, m.otherMed))
            if let statsPath = o.statsPath {
                let row = String(format: "%llu,%d,%d,%.1f,%.3f,%.3f,%.3f,%.2f,%.2f,%d,%.2f,%.2f,%.2f\n",
                                 seed, designed, layout.count, elapsed,
                                 m.caca, m.cacaMin, m.cacaMax, m.rg, m.ifaceMin,
                                 m.contacts, m.hotMed, m.otherMed,
                                 Double(MLX.GPU.peakMemory) / 1_073_741_824)
                if let h = FileHandle(forWritingAtPath: statsPath) {
                    h.seekToEndOfFile(); h.write(Data(row.utf8)); try? h.close()
                }
            }
        }
        let total = Date().timeIntervalSince(batchStart)
        print(String(format: "\n%d designs in %.1f s (%.1f s each)",
                     o.count, total, total / Double(o.count)))
    }

    /// Write both chains, labelled, so the design can be told from the target.
    static func writeComplex(
        _ result: Generator.Result, layout: TokenLayout, vocab: Vocabulary, to path: String
    ) throws {
        var resNames: [String] = [], atomNames: [String] = [], xyz: [Vec3] = []
        var resIDs: [Int] = [], elements: [String] = [], chains: [String] = []
        let letters = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        for (t, token) in layout.tokens.enumerated() {
            let cls = Int(result.aatype[0, t].item(Int32.self))
            guard vocab.classes.indices.contains(cls) else { continue }
            let residue = vocab.classes[cls]
            for slot in 0 ..< Vocabulary.slotCount
            where result.atomMask[0, t, slot].item(Bool.self) {
                let atom = residue.atoms[slot].trimmingCharacters(in: .whitespaces)
                guard !atom.isEmpty else { continue }
                resNames.append(residue.name)
                atomNames.append(atom)
                elements.append(String(atom.prefix(1)))
                resIDs.append(t + 1)
                chains.append(String(letters[token.chainIndex % letters.count]))
                xyz.append(Vec3(result.coordsNM[0, t, slot, 0].item(Float.self),
                                result.coordsNM[0, t, slot, 1].item(Float.self),
                                result.coordsNM[0, t, slot, 2].item(Float.self)))
            }
        }
        var array = AtomArray(resName: resNames, atomName: atomNames, coord: xyz)
        array.resID = resIDs
        array.element = elements
        array.chainID = chains
        try CIFWriter.write(array, to: URL(fileURLWithPath: path), name: "rfd4_binder")
    }

    struct Metrics {
        let caca: Float, cacaMin: Float, cacaMax: Float, rg: Float
        let ifaceMin: Float, contacts: Int, hotMed: Float, otherMed: Float
    }

    /// Geometry per design. These are interface and backbone metrics, not a claim
    /// that anything binds — that is an experimental question.
    static func metrics(
        _ result: Generator.Result, chains: [Range<Int>], hotspotTokens: Set<Int>
    ) -> Metrics {
        let ca = result.xFinal["bb_ca"]![0]
        func geom(_ r: Range<Int>) -> (caca: Float, lo: Float, hi: Float, rg: Float) {
            let c = ca[r.lowerBound ..< r.upperBound]
            let n = c.dim(0)
            guard n > 1 else { return (0, 0, 0, 0) }
            let d = c[1...] - c[..<(n - 1)]
            let step = MLX.sqrt(MLX.sum(d * d, axis: -1)) * 10
            let centred = c - c.mean(axis: 0, keepDims: true)
            let rg = MLX.sqrt(MLX.mean(MLX.sum(centred * centred, axis: -1))) * 10
            return (step.mean().item(Float.self), step.min().item(Float.self),
                    step.max().item(Float.self), rg.item(Float.self))
        }
        let binder = geom(chains[0])
        guard chains.count == 2 else {
            return Metrics(caca: binder.caca, cacaMin: binder.lo, cacaMax: binder.hi,
                           rg: binder.rg, ifaceMin: 0, contacts: 0, hotMed: 0, otherMed: 0)
        }
        let a = ca[chains[0].lowerBound ..< chains[0].upperBound]
        let b = ca[chains[1].lowerBound ..< chains[1].upperBound]
        let diff = a.expandedDimensions(axis: 1) - b.expandedDimensions(axis: 0)
        let dist = MLX.sqrt(MLX.sum(diff * diff, axis: -1)) * 10

        let target = chains[1]
        var hot: [Float] = [], cold: [Float] = []
        for t in target {
            let d = dist[0..., t - target.lowerBound].min().item(Float.self)
            if hotspotTokens.contains(t) { hot.append(d) } else { cold.append(d) }
        }
        func median(_ v: [Float]) -> Float { v.isEmpty ? 0 : v.sorted()[v.count / 2] }
        return Metrics(
            caca: binder.caca, cacaMin: binder.lo, cacaMax: binder.hi, rg: binder.rg,
            ifaceMin: dist.min().item(Float.self),
            contacts: Int(MLX.sum((dist .< 10).asType(.int32)).item(Int32.self)),
            hotMed: median(hot), otherMed: median(cold))
    }
}

enum CLIError: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self { case .message(let m): return m }
    }
}

/// A small deterministic generator for the design-length draw.
///
/// Not the sampler's RNG: the design length is part of the problem statement, and
/// drawing it from the same stream would make the layout depend on how many
/// Gaussians the trajectory happened to consume.
struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
