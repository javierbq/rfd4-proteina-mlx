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
    }

    static func run(_ o: Options) throws {
        let vocab = Vocabulary.unifiedAtom37
        let target = try AtomArray.fromMMCIF(contentsOf: URL(fileURLWithPath: o.target))
        print("target: \(target.count) atoms from \(o.target)")

        // One token per target residue, in file order.
        var residueOrder: [Int] = []
        var residueName: [Int: String] = [:]
        var atomsByResidue: [Int: [(slot: Int, name: String, coord: Vec3)]] = [:]
        let slots = target.atom37Slots(vocab)
        for i in target.indices {
            guard let rid = target.resID?[i], rid >= 0 else { continue }
            let slot = slots[i]
            guard slot >= 0 else { continue }
            if residueName[rid] == nil {
                residueName[rid] = target.resName[i]
                residueOrder.append(rid)
            }
            atomsByResidue[rid, default: []]
                .append((slot, target.atomName[i].trimmingCharacters(in: .whitespaces),
                         target.coord[i]))
        }
        print("target residues: \(residueOrder.count) "
              + "(\(residueOrder.first ?? -1)…\(residueOrder.last ?? -1))")

        // The layout. A draw is supplied rather than left implicit, so the run is
        // reproducible and the design length is reported rather than discovered.
        var rng = SplitMix64(seed: o.seed)
        let layout = try TokenLayout.make(contig: o.contig, draw: { lo, hi in
            o.designLength ?? (lo == hi ? lo : lo + Int(rng.next() % UInt64(hi - lo + 1)))
        })
        let designed = layout.designedMask().filter { $0 }.count
        print("layout: \(layout.count) tokens — \(designed) designed, "
              + "\(layout.count - designed) target, breaks at \(layout.chainBreaks())")

        // Hang the target's atoms on the tokens that preserve them.
        var coords = [[Vec3?]](repeating: [Vec3?](repeating: nil, count: Vocabulary.slotCount),
                               count: layout.count)
        var backbone: [(token: Int, slot: Int)] = []
        var hotspots: [(token: Int, slot: Int)] = []
        var knownSequence = [Int?](repeating: nil, count: layout.count)
        let wanted = Set(o.hotspots.map { "\($0.chain)/\($0.residue)/\($0.atom)" })

        for (t, token) in layout.tokens.enumerated() {
            guard let rid = token.inputResidue, let atoms = atomsByResidue[rid] else { continue }
            for atom in atoms {
                coords[t][atom.slot] = atom.coord
                if o.backboneAtoms.contains(atom.name) { backbone.append((t, atom.slot)) }
                if wanted.contains("\(token.chainID)/\(rid)/\(atom.name)") {
                    hotspots.append((t, atom.slot))
                }
            }
            if let name = residueName[rid],
               let channel = DesignProblem.sequenceProbabilityChannels.firstIndex(of: name) {
                knownSequence[t] = channel
            }
        }
        print("conditioning: \(backbone.count) backbone atoms in one rigid group, "
              + "\(hotspots.count) hotspot atoms, "
              + "\(knownSequence.compactMap { $0 }.count) residues of known sequence")
        guard !backbone.isEmpty else {
            throw CLIError.message("no target backbone atoms matched the contig's residues")
        }
        guard hotspots.count == o.hotspots.count else {
            throw CLIError.message("""
                matched \(hotspots.count) of \(o.hotspots.count) hotspot atoms — check the \
                residue numbering of \(o.target)
                """)
        }

        let problem = DesignProblem(
            layout: layout, coords: coords,
            internalGroups: [DesignProblem.CoordinateGroup(members: backbone)],
            hotspots: hotspots, knownSequence: knownSequence)
        let system = problem.system(seed: o.seed)

        let loadStart = Date()
        let generator = try Generator(weightsAt: URL(fileURLWithPath:
            (o.weights as NSString).expandingTildeInPath))
        print("loaded weights in \(String(format: "%.1f", Date().timeIntervalSince(loadStart)))s")

        var lastReport = Date()
        let runStart = Date()
        let result = try generator.generate(system, strategy: o.strategy, seed: o.seed) {
            step, passes in
            if Date().timeIntervalSince(lastReport) > 10 {
                print("  step \(step + 1)/\(ReleasedRecycleStrategy.nsteps)  \(passes) passes")
                lastReport = Date()
            }
        }
        let elapsed = Date().timeIntervalSince(runStart)

        let chains = layout.chainRanges()
        try writeComplex(result, layout: layout, vocab: vocab, to: o.out)
        report(result, chains: chains, elapsed: elapsed, strategy: o.strategy,
               hotspotTokens: Set(hotspots.map(\.token)))
        print("wrote \(o.out)")
    }

    /// Write both chains, labelled, so the design can be told from the target.
    private static func writeComplex(
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

    /// Per-chain geometry, as a sanity line rather than a quality claim.
    private static func report(
        _ result: Generator.Result, chains: [Range<Int>], elapsed: TimeInterval,
        strategy: ReleasedRecycleStrategy, hotspotTokens: Set<Int>
    ) {
        print(String(format: "generated in %.1fs (%@)", elapsed, strategy.rawValue))
        let ca = result.xFinal["bb_ca"]![0]
        for (i, range) in chains.enumerated() {
            let c = ca[range.lowerBound ..< range.upperBound]
            let n = c.dim(0)
            guard n > 1 else { continue }
            let d = c[1...] - c[..<(n - 1)]
            let caca = MLX.sqrt(MLX.sum(d * d, axis: -1)) * 10
            let centred = c - c.mean(axis: 0, keepDims: true)
            let rg = MLX.sqrt(MLX.mean(MLX.sum(centred * centred, axis: -1))) * 10
            print(String(format: "  chain %d (%d residues)  CA-CA %.2f A (min %.2f max %.2f)  "
                         + "Rg %.1f A", i, n, caca.mean().item(Float.self),
                         caca.min().item(Float.self), caca.max().item(Float.self),
                         rg.item(Float.self)))
        }
        // How far apart the two chains sit: a design that ignored the target would
        // drift away from it entirely.
        guard chains.count == 2 else { return }
        let a = ca[chains[0].lowerBound ..< chains[0].upperBound]
        let b = ca[chains[1].lowerBound ..< chains[1].upperBound]
        let diff = a.expandedDimensions(axis: 1) - b.expandedDimensions(axis: 0)
        let dist = MLX.sqrt(MLX.sum(diff * diff, axis: -1)) * 10
        print(String(format: "  interface: closest CA-CA %.2f A, %d pairs within 10 A",
                     dist.min().item(Float.self),
                     MLX.sum((dist .< 10).asType(.int32)).item(Int32.self)))

        // Did it engage the site it was told to? A binder that merely touches the
        // target somewhere is not the same as one that found the hotspots, and the
        // interface numbers above cannot tell the two apart. The comparison is
        // against the rest of the target, so "close" means close *relative to* the
        // residues it was not asked to engage.
        guard !hotspotTokens.isEmpty else { return }
        let target = chains[1]
        var hot: [Float] = [], cold: [Float] = []
        for t in target {
            let d = dist[0..., t - target.lowerBound].min().item(Float.self)
            if hotspotTokens.contains(t) { hot.append(d) } else { cold.append(d) }
        }
        func median(_ v: [Float]) -> Float { v.sorted()[v.count / 2] }
        print(String(format: "  hotspots: %d residues, closest CA %.1f A (median)  —  "
                     + "rest of target %.1f A",
                     hot.count, median(hot), median(cold)))
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
