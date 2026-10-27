import Foundation
import MLX
import MLXRandom
import RFD4Core

/// A conditioned design: a token layout plus whatever target it is built against.
///
/// This is the piece that turns a request — "design 50 to 120 residues against this
/// structure, engaging these three sidechains" — into the arrays the model reads.
/// Everything it produces is derived from the request; nothing is left at a schema
/// default, because the defaults are not neutral. Measured against upstream's own
/// shipped PD-L1 recipe, a binder lights up 17 of the 35 feature creators where an
/// unconditioned monomer lights 13, and the four that differ are exactly the ones
/// assembled here.
public struct DesignProblem {

    /// A rigid group of target atoms whose shape is given but whose pose is free.
    ///
    /// This is how a protein binder's target enters the model, and it is worth being
    /// precise about what "internal" means: the group's coordinates are centred on
    /// their own centroid and then **randomly rotated**. The model is told the
    /// target's shape and must place it. Supplying the input coordinates unrotated
    /// would be a valid frame too, but a reproducible run should name its rotation
    /// rather than inherit one.
    public struct CoordinateGroup {
        /// `[(token, slot)]` — which atom37 entries belong to this group.
        public var members: [(token: Int, slot: Int)]
        public init(members: [(token: Int, slot: Int)]) { self.members = members }
    }

    public var layout: TokenLayout
    /// `[n][37]` atom coordinates in Angstrom; `nil` where the slot holds no atom.
    public var coords: [[Vec3?]]
    /// Rigid groups whose internal geometry conditions the design.
    public var internalGroups: [CoordinateGroup]
    /// `[(token, slot)]` to pin in absolute space instead. Mutually exclusive with
    /// membership of an internal group — upstream asserts the two modes never overlap.
    public var absoluteAtoms: [(token: Int, slot: Int)]
    /// `[(token, slot)]` the design should engage.
    public var hotspots: [(token: Int, slot: Int)]
    /// Residue class per token for the tokens whose sequence is known; `nil` elsewhere.
    /// Indices are into `sequenceProbabilityChannels`.
    public var knownSequence: [Int?]
    /// `is_in_context_conditioning`: tokens that are context rather than generated
    /// output. **All false for a binder** — upstream's own batch for the shipped
    /// recipe has it empty, so the target is generated too and held in place by its
    /// conditioning. It exists here because partial diffusion is what sets it, and
    /// a silent all-ones emit mask would be wrong the moment that lands.
    public var isContext: [Bool]
    /// `supports_local_latent`: tokens the autoencoder's latent alphabet represents.
    /// True for canonical polymer residues, which is the whole current scope; an
    /// atomized token or a ligand would be false.
    public var supportsLocalLatent: [Bool]

    /// The 28 sequence-probability channels, in `SEQUENCE_PROBABILITY_CONDITIONS`
    /// order: the 20 amino acids alphabetically, then RNA, then DNA.
    ///
    /// The order is load-bearing, and the obvious alternative is wrong in a way that
    /// is hard to see. The factory projects one bias-free matrix across the whole
    /// concatenation, so a permuted channel list misaligns every weight after the
    /// first difference while each individual value still reads correctly. Sorting
    /// these as strings interleaves the nucleotides into the amino acids — `A`,
    /// `ALA`, `ARG`, `ASN`, `ASP`, `C`, … — and costs 8% of the sequence
    /// representation, which is exactly how this list was first written.
    public static let sequenceProbabilityChannels = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "A", "C", "G", "U",
        "DA", "DC", "DG", "DT",
    ]

    public init(
        layout: TokenLayout,
        coords: [[Vec3?]],
        internalGroups: [CoordinateGroup] = [],
        absoluteAtoms: [(token: Int, slot: Int)] = [],
        hotspots: [(token: Int, slot: Int)] = [],
        knownSequence: [Int?],
        isContext: [Bool]? = nil,
        supportsLocalLatent: [Bool]? = nil
    ) {
        self.layout = layout
        self.coords = coords
        self.internalGroups = internalGroups
        self.absoluteAtoms = absoluteAtoms
        self.hotspots = hotspots
        self.knownSequence = knownSequence
        self.isContext = isContext ?? [Bool](repeating: false, count: layout.count)
        self.supportsLocalLatent = supportsLocalLatent
            ?? [Bool](repeating: true, count: layout.count)
    }

    // MARK: - validation

    /// Upstream's `ValidateBatchInvariants` polymerization limit, in Angstrom.
    public static let polymerizationLimit: Float = 2.0

    /// Reject a contig that would bond two residues the input does not join.
    ///
    /// Tokens inside one chain run are bonded consecutively — upstream renumbers each
    /// run from 1 and infers a peptide bond between neighbours, which is why
    /// `TokenLayout` can build connectivity from index alone. The consequence is that
    /// `"A17-20, A30-33"` asks for a bond between residue 20 and residue 30, and
    /// upstream refuses it:
    ///
    ///     [bond_length] A_1:THR4:C -- A_1:VAL5:N = 24.61 Å (limit 2.0 Å).
    ///     Check source coordinates and residue numbering; preserve missing-residue
    ///     gaps instead of bonding across them.
    ///
    /// Without this check the port accepts that contig and designs against a chain
    /// with a 24 Å peptide bond in its connectivity — silently, since every feature
    /// is well-formed and only the chemistry is impossible. The user wanted a `/0`.
    public func validate() throws {
        let slots = Vocabulary.slotCount
        let (nitrogen, carbon) = (0, 2)                  // atom37: N, CA, C, CB, O
        guard layout.count > 1 else { return }

        for i in 1 ..< layout.count {
            let (prev, here) = (layout.tokens[i - 1], layout.tokens[i])
            // Only a pair the layout actually bonds, and only where the input gives
            // both atoms — a designed token has no coordinates to check.
            guard prev.chainIndex == here.chainIndex,
                  let c = coords[i - 1][carbon], let n = coords[i][nitrogen] else { continue }
            let d = ((c.x - n.x) * (c.x - n.x) + (c.y - n.y) * (c.y - n.y)
                     + (c.z - n.z) * (c.z - n.z)).squareRoot()
            guard d.isFinite, d > Self.polymerizationLimit else { continue }
            throw DesignProblemError.impossibleBond(
                token: i, chainID: here.chainID,
                upstreamResidue: prev.inputResidue, downstreamResidue: here.inputResidue,
                distance: d)
        }
    }

    // MARK: - assembly

    /// Build the static system and its conditioning payload.
    ///
    /// - Parameter seed: pins the rotation applied to each internal-coordinate group.
    public func system(seed: UInt64 = 0) -> Generator.System {
        let n = layout.count
        let slots = Vocabulary.slotCount

        func full(_ v: Float, _ shape: [Int]) -> MLXArray { MLX.full(shape, values: MLXArray(v)) }
        func array2(_ rows: [[Int]]) -> MLXArray {
            MLXArray(rows.flatMap { $0.map(Float.init) }, [1, rows.count, rows.count])
        }

        let groups = layout.chainGroupIDs().map(Float.init)
        let pathLength = array2(layout.tokenBondPathLength())
        let ordering = array2(layout.canonicalPolymerOrdering())

        // atom37 presence, straight from which coordinates exist.
        var atom37 = [Float](repeating: 0, count: n * slots)
        for t in 0 ..< n {
            for s in 0 ..< slots where coords[t][s] != nil { atom37[t * slots + s] = 1 }
        }

        // Coordinate groups: -1 unknown, 0 absolute, 1... per internal group.
        var rawGroups = [Int](repeating: CoordGroups.unknown, count: n * slots)
        for (token, slot) in absoluteAtoms { rawGroups[token * slots + slot] = CoordGroups.knownAbsolute }
        for (i, group) in internalGroups.enumerated() {
            for m in group.members { rawGroups[m.token * slots + m.slot] = i + 1 }
        }

        // Internal coordinates: each group centred on its own centroid, then rotated.
        var key = MLXRandom.key(seed)
        var internalCoords = [Float](repeating: .nan, count: n * slots * 3)
        var anyInternal = false
        for group in internalGroups {
            let points = group.members.compactMap { coords[$0.token][$0.slot] }
            guard points.count == group.members.count, !points.isEmpty else { continue }
            anyInternal = true
            let cx = points.reduce(Float(0)) { $0 + $1.x } / Float(points.count)
            let cy = points.reduce(Float(0)) { $0 + $1.y } / Float(points.count)
            let cz = points.reduce(Float(0)) { $0 + $1.z } / Float(points.count)
            let keys = MLXRandom.split(key: key, into: 2)
            key = keys[0]
            let r = Self.randomRotation(key: keys[1])
            for (m, p) in zip(group.members, points) {
                let v = (p.x - cx, p.y - cy, p.z - cz)
                let base = (m.token * slots + m.slot) * 3
                internalCoords[base + 0] = r[0] * v.0 + r[1] * v.1 + r[2] * v.2
                internalCoords[base + 1] = r[3] * v.0 + r[4] * v.1 + r[5] * v.2
                internalCoords[base + 2] = r[6] * v.0 + r[7] * v.1 + r[8] * v.2
            }
        }

        var absolute = [Float](repeating: .nan, count: n * slots * 3)
        for (token, slot) in absoluteAtoms {
            guard let p = coords[token][slot] else { continue }
            let base = (token * slots + slot) * 3
            absolute[base + 0] = p.x; absolute[base + 1] = p.y; absolute[base + 2] = p.z
        }

        var hotspot = [Float](repeating: 0, count: n * slots)
        for (token, slot) in hotspots { hotspot[token * slots + slot] = 1 }

        // Sequence probabilities: a one-hot where the sequence is known, NaN where it
        // is not. NaN is the point — it is what distinguishes "may not be this
        // residue" from "unconstrained", which a probability of zero cannot express.
        let k = Self.sequenceProbabilityChannels.count
        var seqProb = [[Float]](repeating: [Float](repeating: .nan, count: n), count: k)
        for (t, known) in knownSequence.enumerated() {
            guard let known else { continue }
            for c in 0 ..< k { seqProb[c][t] = c == known ? 1 : 0 }
        }

        let emitBB = (0 ..< n).map { isContext[$0] ? Float(0) : 1 }
        let emitLL = (0 ..< n).map { emitBB[$0] * (supportsLocalLatent[$0] ? 1 : 0) }

        let conditioning = Featurizer.Batch.Conditioning(
            internalCoords: anyInternal
                ? MLXArray(internalCoords, [1, n, slots, 3]) : nil,
            absoluteCoords: absoluteAtoms.isEmpty
                ? nil : MLXArray(absolute, [1, n, slots, 3]),
            coordGroups: MLXArray(rawGroups.map(Float.init), [1, n, slots]),
            coordGroupsDensified: MLXArray(
                CoordGroups.densified(rawGroups).map(Float.init), [1, n, slots]),
            sequenceProbabilities: seqProb.map { MLXArray($0, [1, n]) },
            hotspot: MLXArray(hotspot, [1, n, slots]),
            atom37Mask: MLXArray(atom37, [1, n, slots]))

        return Generator.System(
            tokenCount: n,
            tokenType: full(0, [1, n]),
            paddingMask: full(1, [1, n]),
            cathCode: full(-1, [1, n, 3]),
            coordGroup: MLXArray(rawGroups.map(Float.init), [1, n, slots]),
            indexGroup: MLXArray(groups, [1, n]),
            moleculeIID: MLXArray(groups, [1, n]),
            moleculeEntityID: MLXArray(groups, [1, n]),
            bondOrder: full(0, [1, n, n]),
            tokenBondPathLength: pathLength,
            canonicalPolymerOrdering: ordering,
            // `compute_flow_emit_masks`: bb_ca emits where padding & ~context, and
            // local_latents additionally where supports_local_latent.
            //
            // For a binder both reduce to all-ones, because upstream's own batch has
            // `is_in_context_conditioning` empty — the target is generated too and
            // held in place by its conditioning, not by a mask. Writing the formula
            // rather than the all-ones it currently evaluates to is the difference
            // between a contract and a coincidence: partial diffusion sets context,
            // and an atomized token clears supports_local_latent.
            emitBBCA: MLXArray(emitBB, [1, n]),
            emitLocalLatents: MLXArray(emitLL, [1, n]),
            conditioning: conditioning)
    }

    /// Build the system, refusing a contig upstream would refuse.
    public func validatedSystem(seed: UInt64 = 0) throws -> Generator.System {
        try validate()
        return system(seed: seed)
    }

    /// A uniformly random rotation matrix, row-major, from four Gaussians.
    ///
    /// Shoemake's quaternion construction: a normalised 4-vector of independent
    /// normals is uniform on the unit 3-sphere, which maps to the Haar measure on
    /// SO(3). The obvious alternative — three random Euler angles — is *not*
    /// uniform, and clusters near the poles.
    static func randomRotation(key: MLXArray) -> [Float] {
        let q = MLXRandom.normal([4], key: key).asArray(Float.self)
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).squareRoot()
        let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)
        return [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ]
    }
}

public enum DesignProblemError: Error, CustomStringConvertible {
    /// A contig asked for a peptide bond the input's coordinates cannot support.
    case impossibleBond(token: Int, chainID: String, upstreamResidue: Int?,
                        downstreamResidue: Int?, distance: Float)

    public var description: String {
        switch self {
        case .impossibleBond(let token, let chain, let up, let down, let d):
            let pair = [up, down].compactMap { $0.map(String.init) }
            let named = pair.count == 2 ? " (\(chain)\(pair[0]) → \(chain)\(pair[1]))" : ""
            return String(
                format: "contig bonds tokens %d and %d%@ at %.2f A, over the %.1f A "
                + "polymerization limit — the input does not join these residues. "
                + "Insert a `/0` to split them into separate chains, or use a contig "
                + "that preserves the gap.",
                token - 1, token, named, d, Self.limit)
        }
    }

    private static let limit = DesignProblem.polymerizationLimit
}
