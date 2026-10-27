import Foundation
import MLX
import RFD4Core

/// Builds the three things the trunk consumes — `sInit`, `zInit` and `cond` — from a
/// batch, every sampling step.
///
/// All three depend on the step: `sInit` and `zInit` on the noised state `x_t`,
/// `zInit` and `cond` on the time `t`. Only `zInit`'s x_t dependence is subtle —
/// its single x_t feature is binned pairwise **distances**, which are
/// translation-invariant, so a probe that shifts every coordinate uniformly wrongly
/// concludes the pair track is static.
///
/// Each factory concatenates its creators' outputs **in the config's order** and
/// applies one bias-free projection across the whole width. Features that are inert
/// for the current regime still occupy their columns, so they are emitted as
/// explicit zero blocks rather than skipped: dropping one silently misaligns every
/// weight after it.
public struct Featurizer {

    /// One entry in a factory's concatenation: its width, and whether this port
    /// computes it or fills zeros.
    public struct Slot: Sendable {
        public let name: String
        public let width: Int
        public let implemented: Bool

        public init(_ name: String, _ width: Int, implemented: Bool = false) {
            self.name = name
            self.width = width
            self.implemented = implemented
        }
    }

    /// `feats_seq`, in order. Widths are the creators' declared dims.
    public static let seqSlots: [Slot] = [
        Slot("xt_bb_ca", 3, implemented: true),
        Slot("xt_local_latents", 32, implemented: true),
        Slot("token_type", 32, implemented: true),
        Slot("sequence_probabilities", 56),
        Slot("cond_a37coors_nm", 148),
        Slot("cond_a37coors_internal_nm", 148),
        Slot("rasa_binned", 37),
        Slot("secondary_structure", 16),
        Slot("hbond_modernized", 74),
        Slot("pistack", 37),
        Slot("pication", 74),
        Slot("saltbridge", 37),
        Slot("hydrophobic", 37),
        Slot("metal_coord", 37),
        Slot("na_ss_state", 2),
        Slot("hotspot", 37),
        Slot("atom_charge", 16),
        Slot("atomic_element", 16),
        Slot("coord_group_seq", 37, implemented: true),
        Slot("chirality", 18),
        Slot("chain_centroid_xyz", 3),
    ]

    /// `feats_cond_seq`, in order.
    public static let condSlots: [Slot] = [
        Slot("time_emb_bb_ca", 256, implemented: true),
        Slot("time_emb_local_latents", 256, implemented: true),
        Slot("cath_code", 96, implemented: true),
        // One-hots to a constant under recycle.grad="last"; measured inert.
        Slot("recycle_remaining", 1),
    ]

    /// `feats_pair_repr`, in order.
    public static let pairSlots: [Slot] = [
        Slot("xt_bb_ca_pair_dists", 30, implemented: true),
        Slot("bond_mask", 1),
        Slot("bond_order", 5, implemented: true),
        Slot("is_aromatic", 1),
        Slot("index_group_pair", 1, implemented: true),
        Slot("molecule_idx_pair", 1, implemented: true),
        Slot("molecule_entity_pair", 1, implemented: true),
        Slot("token_bond_path_length", 64, implemented: true),
        Slot("canonical_polymer_ordering", 1),
        Slot("na_ss_pair", 2),
    ]

    /// `feats_pair_cond`, in order.
    public static let pairCondSlots: [Slot] = [
        Slot("time_emb_bb_ca", 256, implemented: true),
        Slot("time_emb_local_latents", 256, implemented: true),
    ]

    // MARK: - batch

    /// The subset of the batch contract the live features read.
    public struct Batch {
        public var xtBBCA: MLXArray            // [b, n, 3]
        public var xtLocalLatents: MLXArray    // [b, n, 32]
        public var tBBCA: Float
        public var tLocalLatents: Float
        public var tokenType: MLXArray         // [b, n]
        public var paddingMask: MLXArray       // [b, n]
        public var cathCode: MLXArray          // [b, n, 3], -1 where absent
        public var coordGroup: MLXArray        // [b, n, 37]
        public var indexGroup: MLXArray        // [b, n]
        public var moleculeIID: MLXArray       // [b, n]
        public var moleculeEntityID: MLXArray  // [b, n]
        public var bondOrder: MLXArray         // [b, n, n]
        public var tokenBondPathLength: MLXArray // [b, n, n]

        public var batchSize: Int { xtBBCA.dim(0) }
        public var tokenCount: Int { xtBBCA.dim(1) }
    }

    // MARK: - modules

    let seqFactory: Features.Factory
    let seqTokenType: Features.Embedding
    let condFactory: Features.Factory
    /// Two plain SwiGLU transitions applied to the conditioning before the trunk
    /// ever sees it (`local_latents_transformer.py:_forward_impl`):
    ///
    ///     c = cond_factory(...)
    ///     c = transition_c_2(transition_c_1(c, seq_mask), seq_mask)
    ///
    /// Easy to miss because they sit in the forward method rather than in the
    /// factory, and omitting them leaves `cond` wrong by a factor of several while
    /// the factory itself still passes parity.
    let transitionC1: Transition
    let transitionC2: Transition
    let fold: Features.FoldEmbedding
    let pairFactory: Features.Factory
    let pairCondFactory: Features.Factory
    let pairAdaLN: AdaptiveLayerNorm

    let pairDistMin: Float, pairDistMax: Float, pairDistDim: Int
    let seqSepDim: Int

    public init(
        _ w: Weights,
        pairDistMin: Float = 0.1, pairDistMax: Float = 3.0, pairDistDim: Int = 30,
        seqSepDim: Int = 64
    ) throws {
        let root = w.scoped("nn")
        seqFactory = try Features.Factory(root.scoped("init_repr_factory"))
        // The token-type embedding lives inside creator 2 of feats_seq.
        seqTokenType = try Features.Embedding(
            root.scoped("init_repr_factory.feat_creators.2.embedding"))
        condFactory = try Features.Factory(root.scoped("cond_factory"))
        transitionC1 = try Transition(root.scoped("transition_c_1"))
        transitionC2 = try Transition(root.scoped("transition_c_2"))
        fold = try Features.FoldEmbedding(root.scoped("cond_factory.feat_creators.2"))
        pairFactory = try Features.Factory(root.scoped("pair_repr_builder.init_repr_factory"))
        pairCondFactory = try Features.Factory(root.scoped("pair_repr_builder.cond_factory"))
        pairAdaLN = try AdaptiveLayerNorm(root.scoped("pair_repr_builder.adaln"), kind: .rms)
        self.pairDistMin = pairDistMin
        self.pairDistMax = pairDistMax
        self.pairDistDim = pairDistDim
        self.seqSepDim = seqSepDim
    }

    // MARK: - factories

    /// `init_repr_factory` → `[b, n, tokenDim]`, before the caller applies the mask.
    public func sequenceFeatures(_ b: Batch) -> MLXArray {
        let n = b.tokenCount, bs = b.batchSize
        var parts: [MLXArray] = []
        for slot in Self.seqSlots {
            switch slot.name {
            case "xt_bb_ca": parts.append(b.xtBBCA)
            case "xt_local_latents": parts.append(b.xtLocalLatents)
            case "token_type": parts.append(seqTokenType(b.tokenType))
            case "coord_group_seq": parts.append(b.coordGroup)
            default: parts.append(Features.inert([bs, n, slot.width]))
            }
        }
        return seqFactory(parts)
    }

    /// `cond_factory` alone, **before** the two transitions — the factory's own
    /// output, which is what a per-factory fixture captures.
    public func conditioningFactory(_ b: Batch) -> MLXArray {
        let n = b.tokenCount, bs = b.batchSize
        // The two time embeddings are per-example; the factory's other features are
        // per-token, so they are broadcast across n before concatenation.
        let tBB = MLX.broadcast(Features.timeEmbedding(b.tBBCA, dim: 256)
            .expandedDimensions(axis: 1), to: [bs, n, 256])
        let tLL = MLX.broadcast(Features.timeEmbedding(b.tLocalLatents, dim: 256)
            .expandedDimensions(axis: 1), to: [bs, n, 256])

        var parts: [MLXArray] = []
        for slot in Self.condSlots {
            switch slot.name {
            case "time_emb_bb_ca": parts.append(tBB)
            case "time_emb_local_latents": parts.append(tLL)
            case "cath_code": parts.append(fold(b.cathCode))
            default: parts.append(Features.inert([bs, n, slot.width]))
            }
        }
        return condFactory(parts)
    }

    /// The conditioning the trunk actually receives: the factory output passed
    /// through `transition_c_1` and `transition_c_2`.
    public func conditioning(_ b: Batch) -> MLXArray {
        let c = conditioningFactory(b)
        return transitionC2(transitionC1(c, mask: b.paddingMask), mask: b.paddingMask)
    }

    /// `pair_repr_builder` → `[b, n, n, pairDim]`.
    public func pairRepresentation(_ b: Batch) -> MLXArray {
        let n = b.tokenCount, bs = b.batchSize
        var parts: [MLXArray] = []
        for slot in Self.pairSlots {
            switch slot.name {
            case "xt_bb_ca_pair_dists":
                parts.append(Features.binPairwiseDistances(
                    b.xtBBCA, minDist: pairDistMin, maxDist: pairDistMax, dim: pairDistDim))
            case "bond_order":
                parts.append(Features.oneHot(b.bondOrder, classes: slot.width))
            case "index_group_pair": parts.append(Features.samenessPair(b.indexGroup))
            case "molecule_idx_pair": parts.append(Features.samenessPair(b.moleculeIID))
            case "molecule_entity_pair": parts.append(Features.samenessPair(b.moleculeEntityID))
            case "token_bond_path_length":
                // bin 0 is self / disconnected / cross-group, the last bin is overflow.
                let clamped = MLX.minimum(b.tokenBondPathLength, Float(seqSepDim - 1))
                parts.append(Features.oneHot(clamped, classes: seqSepDim))
            default: parts.append(Features.inert([bs, n, n, slot.width]))
            }
        }
        let repr = pairFactory(parts)

        // Conditioning here is `compact`: one [b, dimCond] vector per example, which
        // the adaptive norm broadcasts across both spatial axes.
        var condParts: [MLXArray] = []
        for slot in Self.pairCondSlots {
            switch slot.name {
            case "time_emb_bb_ca":
                condParts.append(Features.timeEmbedding(b.tBBCA, dim: 256))
            case "time_emb_local_latents":
                condParts.append(Features.timeEmbedding(b.tLocalLatents, dim: 256))
            default: condParts.append(Features.inert([bs, slot.width]))
            }
        }
        let cond = pairCondFactory(condParts)
        let pairMask = b.paddingMask.expandedDimensions(axis: 2)
            * b.paddingMask.expandedDimensions(axis: 1)
        return pairAdaLN(repr, cond: cond, mask: pairMask)
    }

    /// All three trunk inputs for one sampling step.
    public func callAsFunction(_ b: Batch) -> (sInit: MLXArray, zInit: MLXArray, cond: MLXArray) {
        let mask = b.paddingMask.expandedDimensions(axis: -1)
        return (sequenceFeatures(b) * mask, pairRepresentation(b), conditioning(b))
    }

    /// Declared width of each factory's concatenation, for checking against the
    /// projection the checkpoint actually carries.
    public static func width(_ slots: [Slot]) -> Int { slots.reduce(0) { $0 + $1.width } }
}
