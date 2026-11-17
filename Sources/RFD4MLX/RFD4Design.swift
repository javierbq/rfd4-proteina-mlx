import Foundation
import MLX
import RFD4Core

/// Conditioned binder design from an IN-MEMORY target, for an application host.
///
/// ## Why this exists beside `BinderCommand`
///
/// The CLI reads an mmCIF, so it can hand the contig a file's own chain ids and residue
/// numbers. A host has neither: it has a selection the user made, as an ordered array of
/// residues with whatever chain id and numbering the session happens to carry — including
/// insertion codes, which no contig grammar can express. Routing that through a temporary
/// CIF would mean a second parse, and a second parse is a second chance to disagree about
/// which residues the user picked.
///
/// So this takes the array directly and owns the three translations the CLI gets for
/// free:
///
/// 1. **Numbering.** The target is renumbered 1..k per run internally. The host's own
///    identifiers never enter the model, and `Result` is index-aligned to the array that
///    came in, so the host puts them back.
/// 2. **Gaps.** Consecutive residues are joined only where a real peptide bond is
///    present. See ``runs(in:)`` — this is the piece that is wrong by default, and wrong
///    silently.
/// 3. **Frame.** The model places the target itself, from a random rotation, so the
///    output is in an arbitrary frame. ``run(_:target:options:)`` superposes the rebuilt
///    target back onto the supplied one and carries the design through that motion, so
///    what comes out sits beside the structure it was designed against.
///
/// ## What it does NOT do
///
/// No I/O, no weight loading, no file format. The caller supplies a loaded ``Generator``
/// and gets coordinates back. That keeps this testable without a 871 MB pack and leaves
/// the host free to write PDB, mmCIF or nothing at all.
public enum RFD4Design {

    // MARK: - Input

    public struct Atom: Sendable, Equatable {
        public let name: String
        /// Angstrom, in the caller's own frame.
        public let xyz: Vec3

        public init(name: String, xyz: Vec3) {
            self.name = name
            self.xyz = xyz
        }
    }

    /// One target residue. Identity is its POSITION in the array the caller passes;
    /// nothing here carries a chain id or a residue number, because nothing downstream
    /// may depend on one.
    public struct TargetResidue: Sendable, Equatable {
        /// Three-letter name, one of the canonical twenty.
        public let name: String
        public let atoms: [Atom]

        public init(name: String, atoms: [Atom]) {
            self.name = name
            self.atoms = atoms
        }
    }

    public struct Options: Sendable {
        /// Residues to generate.
        public var designLength: Int
        /// POSITIONS in the target array of the interface residues to engage.
        ///
        /// Positions rather than residue numbers, for the reason the type doc gives:
        /// there are no residue numbers at this boundary. Empty is legal and means
        /// unguided placement, which the featurizer supports directly.
        public var hotspots: [Int] = []
        public var seed: UInt64 = 0
        public var strategy: ReleasedRecycleStrategy = .thinkingMedium
        /// `(step, trunkPasses)`, once per sampling step.
        public var onStep: (@Sendable (Int, Int) -> Void)?
        /// Polled once per sampling step. Throws ``GenerationCancelled`` when true.
        public var shouldCancel: (@Sendable () -> Bool)?

        public init(designLength: Int) {
            self.designLength = designLength
        }
    }

    // MARK: - Output

    /// Geometry of one design. **These are geometry, not a claim that anything binds.**
    /// There is no confidence head in this model; a design that passes every number here
    /// is geometrically sane, which is not designable and is certainly not a binder.
    public struct Stats: Sendable {
        /// Mean, min and max distance between consecutive designed CA atoms. An ideal
        /// peptide is 3.80 A.
        public let designCACAMeanA: Double
        public let designCACAMinA: Double
        public let designCACAMaxA: Double
        /// Percentage of consecutive designed CA-CA distances inside [3.6, 4.0] A.
        public let backboneValidPct: Double
        /// Compactness of the designed chain over its CA atoms.
        public let radiusOfGyrationA: Double
        /// Closest CA-CA approach between the designed chain and the target. NEITHER
        /// DIRECTION IS BETTER — far above is a chain floating off the surface, far
        /// below is a clash.
        ///
        /// `nil` when neither side presents a CA to measure between, for the same
        /// reason ``hotspotMinA`` is optional: zero here does not mean "touching", it
        /// means nothing was measured, and a host that renders it reports a catastrophic
        /// clash in a design that has none.
        public let interfaceMinA: Double?
        /// Designed-CA to target-CA pairs closer than 8 A.
        public let contactsUnder8A: Int
        /// Closest approach from the designed chain to any hotspot CA, or `nil` on an
        /// unguided run. Nil rather than zero: upstream-style `isFinite ? x : 0` would
        /// report the best possible score for a thing that was never measured.
        public let hotspotMinA: Double?
        /// RMSD between the target the model rebuilt and the target it was given, after
        /// superposition.
        ///
        /// **This is a different quantity from RFdiffusion3's target drift, and it is
        /// not interchangeable with it.** There the target is masked out of the noise
        /// and reproduced atom-for-atom, so any movement at all is a defect. Here the
        /// target is GENERATED — held only by its conditioning, in a frame the model
        /// chose — so a small non-zero value is the expected result and the number
        /// measures how faithfully the conditioning was honoured.
        public let targetRebuildRMSDA: Double
    }

    public struct Result: Sendable {
        /// One-letter sequence of the generated chain.
        public let designSequence: String
        /// Generated residues in order, all-atom, **in the caller's frame**.
        public let designResidues: [[Atom]]
        /// The target as the model rebuilt it, in the caller's frame and index-aligned
        /// to the input array. Returned for inspection — the caller should emit its OWN
        /// target coordinates, which are exact, rather than these.
        public let rebuiltTarget: [[Atom]]
        /// Tokens the trunk ran over: target residues plus designed residues. What
        /// ``RFD4Budget`` is denominated in.
        public let tokenCount: Int
        public let stats: Stats
    }

    // MARK: - Errors

    public enum Error: Swift.Error, CustomStringConvertible, LocalizedError {
        case emptyTarget
        case badDesignLength(Int)
        case hotspotOutOfRange(index: Int, targetCount: Int)
        case unknownResidue(position: Int, name: String)
        case residueWithoutAtoms(position: Int)
        case tooManyChainBreaks(runs: Int, limit: Int)
        case noBackboneAtoms
        case noHotspotAtoms(position: Int, name: String)
        case designLengthMismatch(expected: Int, produced: Int)
        case frameNotRecovered

        public var description: String {
            switch self {
            case .emptyTarget:
                return "a design needs a target, and this one carries no residues"
            case .badDesignLength(let n):
                return "design length must be at least 1 residue, got \(n)"
            case .hotspotOutOfRange(let index, let count):
                return "hotspot position \(index) is outside the \(count)-residue target"
            case .unknownResidue(let position, let name):
                return "target residue \(position + 1) is \(name), which this model has"
                     + " no atom template for; the canonical twenty only"
            case .residueWithoutAtoms(let position):
                return "target residue \(position + 1) carries no atoms"
            case .tooManyChainBreaks(let runs, let limit):
                return "the target breaks into \(runs) separate chain segments, over the"
                     + " \(limit) this can express. Each gap in the backbone becomes its"
                     + " own segment, so a selection assembled from many disjoint pieces"
                     + " runs out of chain ids. Select a contiguous region."
            case .noBackboneAtoms:
                return "no target residue carries the backbone atoms (N, CA, C, O) that"
                     + " condition the design, so there is no rigid group to build"
            case .noHotspotAtoms(let position, let name):
                return "hotspot at target position \(position + 1) (\(name)) carries"
                     + " none of the atoms a hotspot can be declared on"
            case .designLengthMismatch(let expected, let produced):
                return "the model produced \(produced) designed residues, not the"
                     + " \(expected) requested"
            case .frameNotRecovered:
                return "the rebuilt target could not be matched to the target supplied,"
                     + " so the frame the design was generated in cannot be recovered"
                     + " and the design cannot be placed"
            }
        }

        public var errorDescription: String? { description }
    }

    // MARK: - Layout

    /// Backbone atoms that carry the target's internal geometry — the rigid group the
    /// design is conditioned on. The same four the shipped recipe uses.
    public static let backboneAtomNames: Set<String> = ["N", "CA", "C", "O"]

    /// Chain ids the target's segments may take, in order. 'A' is deliberately absent:
    /// the designed run names no input chain, so `Contig.chainRuns` gives it the first
    /// SYNTHETIC id not claimed by a preserved segment — which is 'A' exactly when the
    /// target does not claim it. That is what makes the designed chain come out as A and
    /// the target as B, matching the CLI's own output without either side coordinating.
    public static let targetChainIDs = Array("BCDEFGHIJKLMNOPQRSTUVWXYZ").map(String.init)

    /// Contiguous stretches of `target`, split wherever the backbone is not bonded.
    ///
    /// **This is the piece that is wrong by default.** A contig that names one unbroken
    /// range over a target with an unmodelled loop asks the model to bond the residues on
    /// either side of the gap, and `DesignProblem.validate()` refuses it by measuring
    /// exactly that bond — "preserve missing-residue gaps instead of bonding across
    /// them". Emitting each stretch as its own segment, separated by a chain break, is
    /// what that message asks for.
    ///
    /// Decided GEOMETRICALLY, from the C(i)-N(i+1) distance against
    /// `DesignProblem.polymerizationLimit`, rather than from residue numbering. Two
    /// reasons: the caller's numbering may be anything at all (insertion codes,
    /// renumbered constructs, a selection assembled out of order), and the distance is
    /// the criterion the engine itself applies — so a layout built this way cannot be
    /// one the validator then rejects.
    ///
    /// A missing C or N reads as a BREAK. The alternative is to assume a bond that
    /// nothing in the input evidences, which is the silent-wrong-answer this exists to
    /// prevent; a residue whose backbone is incomplete is in no position to vouch for a
    /// peptide bond.
    public static func runs(in target: [TargetResidue]) -> [Range<Int>] {
        guard !target.isEmpty else { return [] }
        func atom(_ residue: TargetResidue, _ name: String) -> Vec3? {
            residue.atoms.first { $0.name == name }?.xyz
        }
        var out: [Range<Int>] = []
        var start = 0
        for i in 0 ..< (target.count - 1) {
            let bonded: Bool
            if let c = atom(target[i], "C"), let n = atom(target[i + 1], "N") {
                let dx = c.x - n.x, dy = c.y - n.y, dz = c.z - n.z
                bonded = (dx * dx + dy * dy + dz * dz).squareRoot()
                    <= DesignProblem.polymerizationLimit
            } else {
                bonded = false
            }
            if !bonded {
                out.append(start ..< (i + 1))
                start = i + 1
            }
        }
        out.append(start ..< target.count)
        return out
    }

    /// The contig for `target` split into `segments`, designing `length` residues.
    ///
    /// The designed segment comes FIRST, which is a layout choice with a consequence:
    /// `Contig.chainRuns` walks runs in order and the designed run is the one with no
    /// preserved token, so putting it first makes the designed tokens 0..<length and
    /// lets every later index be read without consulting the layout.
    ///
    /// Each target segment gets its OWN input chain id, even though they all came from
    /// one chain in the caller's session. Required rather than tidy: `chainRuns` refuses
    /// two runs that claim the same input chain (`duplicateOutputChain`), so a gapped
    /// target cannot be written as two ranges of one chain at all.
    static func contig(length: Int, segments: [Range<Int>]) -> String {
        var parts = ["\(length)"]
        for (index, segment) in segments.enumerated() {
            parts.append("/0")
            parts.append("\(targetChainIDs[index])1-\(segment.count)")
        }
        return parts.joined(separator: ", ")
    }

    // MARK: - Running one design

    /// Everything that can be refused WITHOUT a loaded model, and the segments the
    /// layout will be built from.
    ///
    /// Separate from ``run(_:target:options:)`` so a host can refuse before paying for
    /// an 871 MB pack load, and so these refusals are testable without one. Every check
    /// here is one the model would NOT make: its featurizer skips a residue it has no
    /// template for while hotspot positions keep counting, and it designs happily
    /// against an empty target.
    @discardableResult
    public static func preflight(target: [TargetResidue],
                                 options: Options) throws -> [Range<Int>] {
        guard !target.isEmpty else { throw Error.emptyTarget }
        guard options.designLength >= 1 else {
            throw Error.badDesignLength(options.designLength)
        }
        for (position, residue) in target.enumerated() {
            guard Vocabulary.standardAA.contains(residue.name) else {
                throw Error.unknownResidue(position: position, name: residue.name)
            }
            guard !residue.atoms.isEmpty else {
                throw Error.residueWithoutAtoms(position: position)
            }
        }
        for index in options.hotspots {
            guard target.indices.contains(index) else {
                throw Error.hotspotOutOfRange(index: index, targetCount: target.count)
            }
        }
        let segments = runs(in: target)
        guard segments.count <= targetChainIDs.count else {
            throw Error.tooManyChainBreaks(runs: segments.count,
                                           limit: targetChainIDs.count)
        }
        return segments
    }

    /// Tokens one design will run over: target residues plus designed residues.
    ///
    /// What ``RFD4Budget`` is denominated in, exposed so a host can size a design
    /// before building anything.
    public static func tokenCount(target: [TargetResidue], designLength: Int) -> Int {
        target.count + Swift.max(0, designLength)
    }

    public static func run(_ generator: Generator, target: [TargetResidue],
                           options: Options) throws -> Result {
        let vocab = Vocabulary.unifiedAtom37
        let segments = try preflight(target: target, options: options)

        // (internal chain, internal residue number) -> position in `target`. The one
        // map between the model's view of the target and the caller's.
        var positionByKey: [String: Int] = [:]
        var residuesInChain: [String: [Int]] = [:]
        for (index, segment) in segments.enumerated() {
            let chain = targetChainIDs[index]
            residuesInChain[chain] = Array(1 ... segment.count)
            for (offset, position) in segment.enumerated() {
                positionByKey["\(chain)|\(offset + 1)"] = position
            }
        }

        let layout = try TokenLayout.make(
            contig: contig(length: options.designLength, segments: segments),
            residuesInChain: { residuesInChain[$0] ?? [] },
            // Fixed length, so the draw is never consulted; it is required by the
            // signature because a contig MAY carry a range and this one does not.
            draw: { lo, _ in lo })

        var coords = [[Vec3?]](
            repeating: [Vec3?](repeating: nil, count: Vocabulary.slotCount),
            count: layout.count)
        var backbone: [(token: Int, slot: Int)] = []
        var hotspotSlots: [(token: Int, slot: Int)] = []
        var knownSequence = [Int?](repeating: nil, count: layout.count)
        var positionByToken = [Int?](repeating: nil, count: layout.count)
        let hotspotPositions = Set(options.hotspots)

        for (token, entry) in layout.tokens.enumerated() {
            guard let number = entry.inputResidue,
                  let position = positionByKey["\(entry.chainID)|\(number)"]
            else { continue }
            let residue = target[position]
            positionByToken[token] = position
            for atom in residue.atoms {
                guard let slot = vocab.slot(residue: residue.name, atom: atom.name)
                else { continue }
                // An unresolved coordinate poisons the whole rigid group's centroid, so
                // it is dropped from the group rather than carried into it — the
                // `not has_nan_coord()` of the shipped recipe's own selection.
                guard atom.xyz.x.isFinite, atom.xyz.y.isFinite, atom.xyz.z.isFinite
                else { continue }
                coords[token][slot] = atom.xyz
                if backboneAtomNames.contains(atom.name) {
                    backbone.append((token, slot))
                }
            }
            if let channel = DesignProblem.sequenceProbabilityChannels
                .firstIndex(of: residue.name) {
                knownSequence[token] = channel
            }
            if hotspotPositions.contains(position) {
                let slots = hotspotAtomSlots(residue: residue, vocabulary: vocab)
                guard !slots.isEmpty else {
                    throw Error.noHotspotAtoms(position: position, name: residue.name)
                }
                for slot in slots { hotspotSlots.append((token, slot)) }
            }
        }
        guard !backbone.isEmpty else { throw Error.noBackboneAtoms }

        let problem = DesignProblem(
            layout: layout, coords: coords,
            // ONE group for the whole target, across every segment. The chain breaks
            // say the backbone is not bonded across a gap; they do not say the two
            // halves may move relative to each other, and they must not — a gapped
            // domain is one rigid body whose loop is simply unmodelled.
            internalGroups: [DesignProblem.CoordinateGroup(members: backbone)],
            hotspots: hotspotSlots, knownSequence: knownSequence)
        let system = try problem.validatedSystem(seed: options.seed)

        let produced = try generator.generate(
            system, strategy: options.strategy, seed: options.seed,
            onStep: options.onStep.map { callback in { step, passes in callback(step, passes) } },
            shouldCancel: options.shouldCancel)

        return try assemble(produced, layout: layout, target: target,
                            positionByToken: positionByToken,
                            hotspots: options.hotspots,
                            designLength: options.designLength, vocabulary: vocab)
    }

    /// Atom slots a hotspot residue contributes.
    ///
    /// The SIDECHAIN atoms, falling back to CB and then CA. The shipped PD-L1 recipe
    /// names individual sidechain atoms (`A56:CG,OH`), because a hotspot is a claim about
    /// which surface the design should engage and a sidechain tip is what forms that
    /// surface. A host whose UI selects RESIDUES has no atom to name, so the residue-level
    /// reading of the same feature is "every sidechain atom of this residue" — strictly
    /// more signal than one atom and in the same register.
    ///
    /// Backbone atoms are deliberately excluded while any sidechain atom is present. The
    /// fallbacks exist for glycine, which has no sidechain, and for a residue modelled
    /// without one.
    static func hotspotAtomSlots(residue: TargetResidue, vocabulary: Vocabulary) -> [Int] {
        var sidechain: [Int] = []
        var cb: Int?
        var ca: Int?
        for atom in residue.atoms {
            guard let slot = vocabulary.slot(residue: residue.name, atom: atom.name),
                  atom.xyz.x.isFinite, atom.xyz.y.isFinite, atom.xyz.z.isFinite
            else { continue }
            switch atom.name {
            case "CA": ca = slot
            case "CB": cb = slot; sidechain.append(slot)
            case "N", "C", "O", "OXT": continue
            default: sidechain.append(slot)
            }
        }
        if !sidechain.isEmpty { return sidechain }
        if let cb { return [cb] }
        if let ca { return [ca] }
        return []
    }

    // MARK: - Decoding and placement

    /// Pull the result out of MLX, put it back in the caller's frame, and measure it.
    static func assemble(
        _ produced: Generator.Result, layout: TokenLayout, target: [TargetResidue],
        positionByToken: [Int?], hotspots: [Int], designLength: Int, vocabulary: Vocabulary
    ) throws -> Result {
        let n = layout.count
        let slots = Vocabulary.slotCount
        // Pulled as whole arrays rather than element by element. A 226-token complex is
        // 25k slots, and `item()` per element is a separate synchronisation each time.
        let xyz = produced.coordsNM.asArray(Float.self)
        let classes = produced.aatype.asArray(Int32.self)
        let mask = produced.atomMask.asArray(Bool.self)

        var designResidues: [[Atom]] = []
        var designLetters = ""
        var rebuilt = [[Atom]](repeating: [], count: target.count)
        // Paired, index-wise, for the superposition.
        var mobile: [SIMD3<Double>] = []
        var reference: [SIMD3<Double>] = []

        for token in 0 ..< n {
            let cls = Int(classes[token])
            guard vocabulary.classes.indices.contains(cls) else { continue }
            let residueClass = vocabulary.classes[cls]
            var atoms: [Atom] = []
            for slot in 0 ..< slots where mask[token * slots + slot] {
                let name = residueClass.atoms[slot]
                    .trimmingCharacters(in: .whitespaces)
                guard !name.isEmpty else { continue }
                let base = (token * slots + slot) * 3
                // Nanometres out of the model; Angstrom everywhere a caller looks.
                atoms.append(Atom(name: name,
                                  xyz: Vec3(xyz[base] * 10, xyz[base + 1] * 10,
                                            xyz[base + 2] * 10)))
            }
            if let position = positionByToken[token] {
                rebuilt[position] = atoms
                var supplied: [String: Vec3] = [:]
                for atom in target[position].atoms { supplied[atom.name] = atom.xyz }
                for atom in atoms {
                    guard let original = supplied[atom.name] else { continue }
                    mobile.append(SIMD3(Double(atom.xyz.x), Double(atom.xyz.y),
                                        Double(atom.xyz.z)))
                    reference.append(SIMD3(Double(original.x), Double(original.y),
                                           Double(original.z)))
                }
            } else {
                designResidues.append(atoms)
                designLetters.append(oneLetter(residueClass.name))
            }
        }

        guard designResidues.count == designLength else {
            throw Error.designLengthMismatch(expected: designLength,
                                             produced: designResidues.count)
        }
        guard let (transform, rebuildRMSD) = Superposition.fit(mobile: mobile,
                                                               reference: reference)
        else { throw Error.frameNotRecovered }

        // Everything the caller sees is moved by the SAME transform, so the design's
        // pose relative to the target is exactly the pose the model produced. Fitting
        // the design separately would make it superpose on something, which is a
        // different and much weaker claim.
        let placedDesign = designResidues.map { atoms in
            atoms.map { Atom(name: $0.name, xyz: transform.apply($0.xyz)) }
        }
        let placedTarget = rebuilt.map { atoms in
            atoms.map { Atom(name: $0.name, xyz: transform.apply($0.xyz)) }
        }

        return Result(
            designSequence: designLetters,
            designResidues: placedDesign,
            rebuiltTarget: placedTarget,
            tokenCount: n,
            stats: statistics(design: placedDesign, target: target, hotspots: hotspots,
                              rebuildRMSD: rebuildRMSD))
    }

    /// Geometry over the placed design and the SUPPLIED target.
    ///
    /// The supplied target rather than the rebuilt one, deliberately: those are the
    /// coordinates the caller will emit beside the design, so an interface distance
    /// measured against anything else would describe a complex nobody is going to see.
    static func statistics(design: [[Atom]], target: [TargetResidue], hotspots: [Int],
                           rebuildRMSD: Double) -> Stats {
        func ca(_ atoms: [Atom]) -> SIMD3<Double>? {
            atoms.first { $0.name == "CA" }
                .map { SIMD3(Double($0.xyz.x), Double($0.xyz.y), Double($0.xyz.z)) }
        }
        let designCA = design.compactMap(ca)
        func distance(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
            let d = a - b
            return (d * d).sum().squareRoot()
        }

        var steps: [Double] = []
        for i in 1 ..< Swift.max(designCA.count, 1) {
            steps.append(distance(designCA[i], designCA[i - 1]))
        }
        let mean = steps.isEmpty ? 0 : steps.reduce(0, +) / Double(steps.count)
        let inRange = steps.filter { $0 >= 3.6 && $0 <= 4.0 }.count
        let validPct = steps.isEmpty ? 0 : 100 * Double(inRange) / Double(steps.count)

        var radius = 0.0
        if !designCA.isEmpty {
            let centre = designCA.reduce(SIMD3<Double>.zero, +) / Double(designCA.count)
            let sum = designCA.reduce(0.0) { acc, p in
                let d = p - centre
                return acc + (d * d).sum()
            }
            radius = (sum / Double(designCA.count)).squareRoot()
        }

        var interfaceMin = Double.infinity
        var contacts = 0
        var hotspotMin = Double.infinity
        let hotspotSet = Set(hotspots)
        // Target CA list is compacted, so walk the residues to keep positions aligned
        // with `hotspots`, which index the ORIGINAL array.
        var caByPosition: [Int: SIMD3<Double>] = [:]
        for (position, residue) in target.enumerated() {
            if let point = ca(residue.atoms) { caByPosition[position] = point }
        }
        for a in designCA {
            for (position, b) in caByPosition {
                let d = distance(a, b)
                interfaceMin = Swift.min(interfaceMin, d)
                if d < 8 { contacts += 1 }
                if hotspotSet.contains(position) { hotspotMin = Swift.min(hotspotMin, d) }
            }
        }

        return Stats(
            designCACAMeanA: mean,
            designCACAMinA: steps.min() ?? 0,
            designCACAMaxA: steps.max() ?? 0,
            backboneValidPct: validPct,
            radiusOfGyrationA: radius,
            interfaceMinA: interfaceMin.isFinite ? interfaceMin : nil,
            contactsUnder8A: contacts,
            hotspotMinA: hotspotMin.isFinite ? hotspotMin : nil,
            targetRebuildRMSDA: rebuildRMSD)
    }

    private static let threeToOne: [String: Character] = [
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V",
    ]

    /// One-letter code, or `X` for anything outside the canonical twenty — which the
    /// sequence head can emit, since nothing constrains its argmax to them.
    public static func oneLetter(_ residue: String) -> Character {
        threeToOne[residue] ?? "X"
    }
}
