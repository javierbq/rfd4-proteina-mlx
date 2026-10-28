import Foundation
import MLX
import MLXRandom
import RFD4Core

/// The autoencoder encoder — `partial_autoencoder/encoder.py:EncoderTransformer`.
///
/// The piece partial diffusion needs. The flow model works on `local_latents`, so
/// starting a trajectory from a real structure rather than from noise means first
/// turning that structure into latents, and that is this network's whole job.
///
/// Structurally it is the decoder's twin: twelve layers of the same block, with
/// **no pair track at all** (`use_attn_pair_bias: False`, `update_pair_repr: False`
/// in `nn_ae/nn_130m_unified.yaml`, and no pair parameters anywhere under
/// `autoencoder.encoder` in the checkpoint), and LayerNorm rather than the trunk's
/// RMSNorm. Two things differ.
///
/// The features read the **ground-truth** structure `x1`, not the noised `x_t`:
/// atom37 coordinates absolute and CA-relative, the chain index, incident bond-type
/// counts, and an element-plus-charge embedding per occupied slot. Those are
/// measurements of a structure the caller already has, which is the opposite of
/// everything the flow trunk consumes.
///
/// And the head is **stochastic**. `latent_decoder_mean_n_log_scale` projects to
/// `2 * latentDim` and the halves are a mean and a log-scale, from which the latent
/// is drawn: `z = mean + randn * exp(log_scale)`. Parity is therefore checked
/// against `mean` and `log_scale`, which are deterministic; the draw is matched
/// separately, and the two must never be conflated — a port that compared `z` to `z`
/// would be comparing two different samples and calling the difference an error.
public enum EncoderError: Error, CustomStringConvertible {
    /// An atom37 slot the mask claims but the structure does not model.
    case unresolvedOccupiedSlot(count: Int, token: Int, slot: Int)

    public var description: String {
        switch self {
        case .unresolvedOccupiedSlot(let count, let token, let slot):
            return """
                \(count) occupied atom37 slots have non-finite coordinates, the first \
                at token \(token), slot \(slot). The encoder would return an entirely \
                NaN latent: `coords * mask` is NaN where the mask is 1 and the atom is \
                unmodelled, and attention spreads that to every token in the first \
                layer. Encode a structure whose modelled atoms match its atom37 mask \
                — a generated complex does, a crystal structure with unresolved \
                sidechains does not.
                """
        }
    }
}

public struct Encoder {
    public static let layerCount = 12
    public static let latentDim = 32
    /// `C_CHARGE`: formal charges −4…+6 map to rows 0…10, out of range clamped.
    static let chargeOffset = 4
    static let chargeClasses = 11

    let layers: [TrunkLayer]
    /// Six features, 1587 wide: token type (32), atom37 coordinates absolute (148)
    /// and CA-relative (148), chain index (1), incident bond-type counts (370), and
    /// the element/charge embedding (888). Verified against `linear_out`'s input
    /// dimension, which is the cheapest check that the concatenation is right.
    let seqFactory: Features.Factory
    let tokenTypeEmbedding: Features.Embedding
    let elementEmbedding: Features.Embedding
    let chargeEmbedding: Features.Embedding
    let head: Decoder.LNHead

    public struct Output {
        /// `[b, n, 32]` the latent's mean — what a deterministic encode returns.
        public let mean: MLXArray
        /// `[b, n, 32]` log standard deviation.
        public let logScale: MLXArray
        /// `[b, n, 32]` a draw, or the mean when none was requested.
        public let z: MLXArray
        /// `[b, n]` tokens a latent was emitted for.
        public let emitMask: MLXArray
    }

    public init(
        _ w: Weights,
        heads: Int = 12,
        dimHead: Int = 64,
        eps: Float = 1e-5
    ) throws {
        let root = w.scoped("autoencoder.encoder")
        layers = try (0 ..< Self.layerCount).map {
            try TrunkLayer(root.scoped("transformer_layers", $0),
                           heads: heads, dimHead: dimHead, kind: .layer)
        }
        seqFactory = try Features.Factory(root.scoped("init_repr_factory"), kind: .layer)
        tokenTypeEmbedding = try Features.Embedding(
            root.scoped("init_repr_factory.feat_creators.0.embedding"))
        elementEmbedding = try Features.Embedding(
            root.scoped("init_repr_factory.feat_creators.5.element_embedding"))
        chargeEmbedding = try Features.Embedding(
            root.scoped("init_repr_factory.feat_creators.5.charge_embedding"))
        head = try Decoder.LNHead(root.scoped("latent_decoder_mean_n_log_scale"), eps: eps)
    }

    /// What the encoder measures off a structure the caller already has.
    public struct Structure {
        /// `[b, n, 37, 3]` atom37 coordinates in **nanometres**.
        ///
        /// Upstream's own wrapper derives this (`autoencoder.py:188`), dividing the
        /// batch's Angstrom coordinates by 10 before the encoder ever runs — it is
        /// not a dataset field, and the feature asserts on its presence.
        public var coordsNM: MLXArray
        /// `[b, n, 37]` which slots hold an atom.
        public var atom37Mask: MLXArray
        /// `[b, n]` token type.
        public var tokenType: MLXArray
        /// `[b, n]` chain index, used as a bare scalar feature.
        public var chainIndex: MLXArray
        /// `[b, n, 37, 10]` per-slot counts of incident bond types.
        public var incidentBondTypeCounts: MLXArray
        /// `[b, n, 37]` atomic number per slot (`ground_truth_element`).
        public var element: MLXArray
        /// `[b, n, 37]` formal charge per slot.
        public var charge: MLXArray
        /// `[b, n]` tokens the encoder attends over — `padding & ~context`.
        public var attnMask: MLXArray
        /// `[b, n]` tokens it emits a latent for — `attn & supports_local_latent`.
        public var emitMask: MLXArray

        public init(
            coordsNM: MLXArray, atom37Mask: MLXArray, tokenType: MLXArray,
            chainIndex: MLXArray, incidentBondTypeCounts: MLXArray,
            element: MLXArray, charge: MLXArray,
            attnMask: MLXArray, emitMask: MLXArray
        ) {
            self.coordsNM = coordsNM
            self.atom37Mask = atom37Mask
            self.tokenType = tokenType
            self.chainIndex = chainIndex
            self.incidentBondTypeCounts = incidentBondTypeCounts
            self.element = element
            self.charge = charge
            self.attnMask = attnMask
            self.emitMask = emitMask
        }

        /// Refuse a structure the encoder cannot encode.
        ///
        /// An atom37 slot the mask claims but the structure does not model makes
        /// `coords * mask` NaN — IEEE, not a choice — and attention spreads that to
        /// every token in the first layer, so the encoder returns an entirely NaN
        /// latent. **Upstream does the same**: measured on its own shipped binder
        /// target, where six unresolved atoms in occupied slots give `mean`,
        /// `log_scale` and `z` all NaN across all 54 tokens. Its own partial diffusion
        /// never trips over it because that flow consumes a generated complex, where
        /// every atom the mask claims really exists.
        ///
        /// The forward pass deliberately does **not** call this: it stays faithful, NaN
        /// and all, so parity is a comparison of the same arithmetic. Callers that take
        /// a structure from a user should call it, and the partial-diffusion entry point
        /// does — an all-NaN latent is not a degraded answer, it is no answer, and it
        /// would otherwise surface as a structure file full of NaN coordinates.
        public func validate() throws {
            let unresolved = MLX.logicalAnd(
                MLX.logicalNot(MLX.all(MLX.isFinite(coordsNM), axis: -1)),
                atom37Mask .> 0)
            let count = MLX.sum(unresolved.asType(.int32)).item(Int32.self)
            guard count == 0 else {
                // Name the first one, so the caller can find it in their file.
                let flat = unresolved.reshaped(-1).asArray(Bool.self)
                let first = flat.firstIndex(of: true) ?? 0
                throw EncoderError.unresolvedOccupiedSlot(
                    count: Int(count),
                    token: first / Vocabulary.slotCount,
                    slot: first % Vocabulary.slotCount)
            }
        }
    }

    // MARK: - features

    /// `Atom37NanometersCoorsSeqFeat` — `[b, n, 37 * 4]`.
    ///
    /// Note what this does NOT do, against its conditional cousin: no division by 10,
    /// because `coordsNM` is already in nanometres, and the mask comes from
    /// `atom37_mask` rather than from which coordinates are finite. Reusing
    /// `conditionalAtom37Coords` here would be wrong twice over.
    public static func atom37Coords(
        _ coordsNM: MLXArray, atom37Mask: MLXArray, relative: Bool
    ) -> MLXArray {
        let maskF = atom37Mask.asType(.float32)
        var coors = coordsNM * maskF.expandedDimensions(axis: -1)
        if relative {
            let ca = coors[0..., 0..., 1, 0...].expandedDimensions(axis: 2)
            coors = (coors - ca) * maskF.expandedDimensions(axis: -1)
        }
        let (b, n) = (coordsNM.dim(0), coordsNM.dim(1))
        let flat = coors.reshaped(b, n, Vocabulary.slotCount * 3)
        return MLX.concatenated([flat, maskF], axis: -1)
    }

    /// `IncidentBondTypeCountsSeqFeat` — `[b, n, 37 * 10]`.
    public static func incidentBondTypeCounts(
        _ counts: MLXArray, atom37Mask: MLXArray
    ) -> MLXArray {
        let (b, n) = (counts.dim(0), counts.dim(1))
        let masked = counts.asType(.float32)
            * atom37Mask.asType(.float32).expandedDimensions(axis: -1)
        return masked.reshaped(b, n, Vocabulary.slotCount * counts.dim(3))
    }

    /// `Atom37ElementChargeEmbeddingSeqFeat` — `[b, n, 37 * (8 + 16)]`.
    ///
    /// Both tables are looked up at **every** slot and the result masked afterwards,
    /// rather than gathering only the occupied ones: an unoccupied slot indexes row
    /// 0 of each table and is then multiplied away. Gathering instead would give the
    /// same answer here and a different one the moment row 0 stops being neutral.
    public func elementCharge(_ element: MLXArray, charge: MLXArray,
                              atom37Mask: MLXArray) -> MLXArray {
        let mask = atom37Mask .> 0
        let zero = MLXArray(Int32(0))
        let elementIndex = MLX.where(mask, element.asType(.int32), zero)
        let chargeIndex = MLX.clip(
            MLX.where(mask, charge.asType(.int32), zero) + Int32(Self.chargeOffset),
            min: Int32(0), max: Int32(Self.chargeClasses - 1))

        let encoded = MLX.concatenated(
            [elementEmbedding(elementIndex), chargeEmbedding(chargeIndex)], axis: -1)
        let masked = encoded * atom37Mask.asType(.float32).expandedDimensions(axis: -1)
        let (b, n) = (element.dim(0), element.dim(1))
        return masked.reshaped(b, n, Vocabulary.slotCount * encoded.dim(3))
    }

    /// The encoder's initial sequence representation, in creator order.
    public func sequenceFeatures(_ s: Structure) -> MLXArray {
        seqFactory([
            tokenTypeEmbedding(s.tokenType),
            Self.atom37Coords(s.coordsNM, atom37Mask: s.atom37Mask, relative: false),
            Self.atom37Coords(s.coordsNM, atom37Mask: s.atom37Mask, relative: true),
            s.chainIndex.expandedDimensions(axis: -1).asType(.float32),
            Self.incidentBondTypeCounts(s.incidentBondTypeCounts, atom37Mask: s.atom37Mask),
            elementCharge(s.element, charge: s.charge, atom37Mask: s.atom37Mask),
        ])
    }

    // MARK: - forward

    /// Encode a structure into per-residue latents.
    ///
    /// - Parameter seed: when given, draws `z` from the predicted distribution with
    ///   that seed; otherwise `z` is the mean. The distinction is the point — a
    ///   partial-diffusion run that silently used the mean would be a different
    ///   generative model from the one the weights were trained for, and a parity
    ///   test that compared two independent draws would report an error that is not
    ///   one.
    public func callAsFunction(_ s: Structure, seed: UInt64? = nil) -> Output {
        let n = s.tokenType.dim(1), b = s.tokenType.dim(0)
        var seqs = sequenceFeatures(s) * s.attnMask.expandedDimensions(axis: -1)

        // `cond_factory` carries no parameters — `feats_cond_seq` is empty — so the
        // conditioning entering the transitions is zero, and the transitions are
        // bias-free SwiGLU with no norm, so zero passes through them unchanged. The
        // same reasoning the decoder relies on, and the reason neither needs them.
        let cond = MLX.zeros([b, n, 128])

        for layer in layers {
            seqs = layer(seqs, pairRep: nil, cond: cond, mask: s.attnMask)
        }

        // The emit mask is applied to the full 2 * latentDim BEFORE the split, so it
        // masks the mean and the log-scale alike.
        let flat = head(seqs) * s.emitMask.expandedDimensions(axis: -1)
        let mean = flat[0..., 0..., ..<Self.latentDim]
        let logScale = flat[0..., 0..., Self.latentDim...]

        var z = mean
        if let seed {
            let noise = MLXRandom.normal([b, n, Self.latentDim], key: MLXRandom.key(seed))
            z = mean + noise * MLX.exp(logScale)
        }
        // `ln_z` is Identity in the released config; the mask is not.
        z = z * s.emitMask.expandedDimensions(axis: -1)

        return Output(mean: mean, logScale: logScale, z: z, emitMask: s.emitMask)
    }
}
