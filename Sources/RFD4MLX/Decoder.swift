import Foundation
import MLX
import RFD4Core

/// The autoencoder decoder — `partial_autoencoder/decoder.py:DecoderTransformer`.
///
/// Expands the flow model's per-residue output — a CA position and a 32-dim latent —
/// into all-atom geometry and a residue identity. Twelve layers of the same
/// attention+MLP block the trunk uses, but with **no pair track at all**
/// (`use_attn_pair_bias: False`, `update_pair_repr: False`), so attention carries
/// only the mask, and with **LayerNorm** rather than the trunk's RMSNorm.
///
/// As with `Trunk`, the feature factories that build `seqInit` and `cond` are not
/// part of this type: it takes them as arguments so the network can be driven and
/// verified before the featurizer exists.
public struct Decoder {
    public static let layerCount = 12
    public static let atomSlots = Vocabulary.slotCount   // 37

    let layers: [TrunkLayer]
    let logitHead: LNHead
    let structHead: LNHead
    let presenceHead: LNHead

    /// `[7][147]` and `[147][37]`, derived from the vocabulary rather than read from
    /// the checkpoint. Upstream registers them as buffers computed from the same YAML,
    /// so deriving them here also cross-checks `RFD4Core`'s transcription against the
    /// network that consumes it.
    let allowedResidueTypes: MLXArray
    let residueAtomMask: MLXArray

    /// `Sequential(LayerNorm(dim), Linear(dim, out, bias=False))`.
    public struct LNHead {
        let norm: Norm
        let projW: MLXArray

        init(_ w: Weights, eps: Float) throws {
            norm = Norm(w.scoped("0"), kind: .layer, eps: eps)
            projW = try w("1.weight")
        }

        func callAsFunction(_ s: MLXArray) -> MLXArray { linear(norm(s), projW) }
    }

    public struct Output {
        /// `[b, n, 37, 3]` all-atom coordinates, in nanometres.
        public let coordsNM: MLXArray
        /// `[b, n, 147]` residue-type logits, already masked to the legal set.
        public let seqLogits: MLXArray
        /// `[b, n]` most likely residue class.
        public let aatype: MLXArray
        /// `[b, n, 37]` which slots carry a real atom.
        public let atomMask: MLXArray
        public let atomPresenceLogits: MLXArray
    }

    public init(
        _ w: Weights,
        heads: Int = 12,
        dimHead: Int = 64,
        vocabulary: Vocabulary = .unifiedAtom37,
        eps: Float = 1e-5
    ) throws {
        let root = w.scoped("autoencoder.decoder")
        // LayerNorm, not RMSNorm: the autoencoder was built under a different
        // make_norm setting than the flow trunk.
        layers = try (0 ..< Self.layerCount).map {
            try TrunkLayer(root.scoped("transformer_layers", $0),
                           heads: heads, dimHead: dimHead, kind: .layer)
        }
        logitHead = try LNHead(root.scoped("logit_linear"), eps: eps)
        structHead = try LNHead(root.scoped("struct_linear"), eps: eps)
        presenceHead = try LNHead(root.scoped("atom_presence_linear"), eps: eps)

        let allowed = vocabulary.allowedResidueTypeMask()
        allowedResidueTypes = MLXArray(
            allowed.flatMap { $0.map { $0 ? Float(1) : Float(0) } },
            [allowed.count, vocabulary.tokenCount])
        let atoms = vocabulary.residueAtomMask()
        residueAtomMask = MLXArray(
            atoms.flatMap { $0.map { $0 ? Float(1) : Float(0) } },
            [atoms.count, Vocabulary.slotCount])
    }

    /// - Parameters:
    ///   - seqInit: `[b, n, 768]` initial sequence representation
    ///   - cond: `[b, n, 128]` conditioning
    ///   - attnMask: `[b, n]` tokens the decoder attends over
    ///   - emitMask: `[b, n]` tokens it reconstructs
    ///   - caCoordsNM: `[b, n, 3]` the CA positions the flow model produced
    ///   - tokenType: `[b, n]` integer token types, used to mask illegal residues
    public func callAsFunction(
        seqInit: MLXArray, cond: MLXArray, attnMask: MLXArray,
        emitMask: MLXArray, caCoordsNM: MLXArray, tokenType: MLXArray? = nil
    ) -> Output {
        var s = seqInit * attnMask.expandedDimensions(axis: -1)
        for layer in layers {
            s = layer(s, pairRep: nil, cond: cond, mask: attnMask)
        }

        let emit = emitMask.expandedDimensions(axis: -1)
        var logits = logitHead(s) * emit

        // Restrict each token to the residue classes its type may decode to.
        // An emitted token keeps only its allowed classes; a non-emitted one is
        // left alone, which is upstream's `allowed | ~emit_mask`.
        if let tokenType {
            let allowedPerToken = allowedResidueTypes[tokenType]      // [b, n, 147]
            let keep = (allowedPerToken .> 0) .|| (emit .<= 0)
            logits = MLX.where(keep, logits, MLXArray(-Float.greatestFiniteMagnitude))
        }

        // Relative coordinates: slot 1 is forced to the supplied CA and every other
        // slot is an offset from it. `abs_coors` is false in the released decoder.
        let n = s.dim(1), b = s.dim(0)
        var coords = (structHead(s) * emit).reshaped(b, n, Self.atomSlots, 3)
        let slotIsCA = MLXArray((0 ..< Self.atomSlots).map { $0 == 1 ? Float(0) : Float(1) })
            .reshaped(1, 1, Self.atomSlots, 1)
        coords = coords * slotIsCA + caCoordsNM.expandedDimensions(axis: 2)
        coords = coords * emit.expandedDimensions(axis: -1)

        let aatype = MLX.argMax(logits, axis: -1)
        let aatypeMasked = aatype * emitMask.asType(aatype.dtype)

        let presence = presenceHead(s)
        var predicted = presence .> 0
        // Every emitted token requires its reference atom, whatever the logit says.
        let isSlot1 = MLXArray((0 ..< Self.atomSlots).map { $0 == 1 ? true : false })
            .reshaped(1, 1, Self.atomSlots)
        predicted = predicted .|| isSlot1
        let allowedAtoms = residueAtomMask[aatypeMasked] .> 0
        let atomMask = allowedAtoms .&& (emit .> 0) .&& predicted

        return Output(coordsNM: coords, seqLogits: logits, aatype: aatypeMasked,
                      atomMask: atomMask, atomPresenceLogits: presence)
    }
}
