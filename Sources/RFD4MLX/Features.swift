import Foundation
import MLX
import RFD4Core

/// The feature creators that build the trunk's three inputs.
///
/// Upstream has 35 of them across the three factories. **13 carry signal for an
/// unconditioned canonical-20 protein; the other 22 are identically zero in that
/// regime** and come alive only under conditioning — hotspots, RASA, secondary
/// structure, hydrogen bonds, pi-stacking, chirality and so on. Those 22 still need
/// a slot in the concatenation, because `linear_out` is one matrix over the full
/// width, but they need no implementation until conditioning lands. Each is a
/// measured fact, not an assumption: see `FeatureParityTests`.
public enum Features {

    // MARK: - primitives

    /// Sinusoidal time embedding (`feature_factory.py:get_time_embedding`).
    ///
    /// Note `t` is scaled by `maxPositions` **before** the frequencies are applied,
    /// so this is a positional encoding over `t * 2000` rather than over `t`.
    public static func timeEmbedding(
        _ t: Float, dim: Int, maxPositions: Float = 2000
    ) -> MLXArray {
        let scaled = t * maxPositions
        let halfDim = dim / 2
        let step = log(maxPositions) / Float(halfDim - 1)
        let freqs = (0 ..< halfDim).map { exp(Float($0) * -step) }
        let angles = freqs.map { scaled * $0 }
        let emb = angles.map { sin($0) } + angles.map { cos($0) }
        // Upstream zero-pads an odd dim; the released config uses 256.
        let padded = dim % 2 == 1 ? emb + [0] : emb
        return MLXArray(padded, [1, dim])
    }

    /// One-hot of `values` bucketed by `limits` (`bin_and_one_hot`).
    ///
    /// `d - 1` limits give `d` bins: everything below the first, the intervals
    /// between, and everything above the last. Matches `torch.bucketize`'s default
    /// right-open convention.
    public static func binAndOneHot(_ values: MLXArray, limits: [Float]) -> MLXArray {
        var index = MLX.zeros(like: values)
        for limit in limits {
            index = index + (values .>= limit).asType(values.dtype)
        }
        return oneHot(index, classes: limits.count + 1)
    }

    /// One-hot along a new trailing axis.
    public static func oneHot(_ indices: MLXArray, classes: Int) -> MLXArray {
        let shape = indices.shape + [classes]
        let flat = indices.asType(.int32).reshaped(-1, 1)
        let range = MLXArray(Array(Int32(0) ..< Int32(classes))).reshaped(1, classes)
        return (flat .== range).asType(.float32).reshaped(shape)
    }

    /// Binned pairwise CA distances (`bin_pairwise_distances`).
    ///
    /// This is the feature that makes the pair representation depend on `x_t`, and
    /// it is **translation-invariant** — which is why probing the featurizer by
    /// shifting every coordinate by a constant wrongly suggests the pair track is
    /// static. Perturb randomly, not uniformly.
    public static func binPairwiseDistances(
        _ x: MLXArray, minDist: Float, maxDist: Float, dim: Int
    ) -> MLXArray {
        let diff = x.expandedDimensions(axis: 2) - x.expandedDimensions(axis: 1)
        let dist = MLX.sqrt(MLX.sum(diff * diff, axis: -1))
        return binAndOneHot(dist, limits: linspace(minDist, maxDist, dim - 1))
    }

    /// `a[:, :, None] == a[:, None, :]`, as a `[b, n, n, 1]` float.
    ///
    /// Three of the live pair features are exactly this on different token-level
    /// groupings: index group, molecule id, molecule entity.
    public static func samenessPair(_ a: MLXArray) -> MLXArray {
        let same = a.expandedDimensions(axis: 2) .== a.expandedDimensions(axis: 1)
        return same.asType(.float32).expandedDimensions(axis: -1)
    }

    // MARK: - learned embeddings

    /// `torch.nn.Embedding` — a row lookup.
    public struct Embedding {
        let table: MLXArray

        public init(_ w: Weights, key: String = "weight") throws {
            table = try w(key)
        }

        public func callAsFunction(_ indices: MLXArray) -> MLXArray {
            table[indices.asType(.int32)]
        }
    }

    /// CATH fold conditioning: three embeddings concatenated (`FoldEmbeddingSeqFeat`).
    ///
    /// Each level gets one extra class because null is stored as -1 and shifted up
    /// by one, so an unconditioned run indexes row 0 of each table rather than
    /// falling out of range.
    public struct FoldEmbedding {
        let c: Embedding, a: Embedding, t: Embedding

        public init(_ w: Weights) throws {
            c = try Embedding(w.scoped("embedding_C"))
            a = try Embedding(w.scoped("embedding_A"))
            t = try Embedding(w.scoped("embedding_T"))
        }

        /// - Parameter cath: `[b, n, 3]` CATH codes, `-1` where absent.
        public func callAsFunction(_ cath: MLXArray) -> MLXArray {
            let shifted = cath.asType(.int32) + 1
            return MLX.concatenated([
                c(shifted[.ellipsis, 0]),
                a(shifted[.ellipsis, 1]),
                t(shifted[.ellipsis, 2]),
            ], axis: -1)
        }
    }

    // MARK: - factory

    /// `FeatureFactory`: concatenate every creator's output, then one bias-free
    /// projection to the factory's width, optionally normalised.
    ///
    /// The projection spans the **full** concatenated width including the inert
    /// features, so their zero blocks still consume columns of `linear_out`. A port
    /// that omits them silently misaligns every weight after the first gap.
    public struct Factory {
        let projW: MLXArray
        let norm: Norm?

        public init(_ w: Weights, kind: NormKind = .rms, eps: Float = 1e-5) throws {
            projW = try w("linear_out.weight")
            norm = w.has("ln_out.weight") ? Norm(w.scoped("ln_out"), kind: kind, eps: eps) : nil
        }

        /// - Parameter parts: creator outputs **in upstream's order**.
        public func callAsFunction(_ parts: [MLXArray]) -> MLXArray {
            let cat = MLX.concatenated(parts, axis: -1)
            let out = linear(cat, projW)
            return norm.map { $0(out) } ?? out
        }

        /// Width the projection expects, for checking a concatenation before it runs.
        public var inputWidth: Int { projW.dim(1) }
    }

    // MARK: - conditioning

    /// Conditional all-atom coordinates — `ConditionalAtom37NanometersCoorsSeqFeat`.
    ///
    /// **This is how a target structure enters the model.** Without it the network
    /// has no idea a target exists, which is why every conditioning feature measures
    /// inert on an unconditioned run and why binder design needs exactly this one.
    ///
    /// Two details that are easy to lose and silently halve the signal:
    ///   - the batch carries these in **Angstrom**; the model works in nanometres,
    ///     so they are divided by 10 here.
    ///   - unconditioned positions arrive as **NaN**, not as zeros. The mask is
    ///     derived from which rows are finite, then NaN is replaced with zero. A
    ///     port that trusted zeros would treat "no constraint" as "constrained to
    ///     the origin".
    ///
    /// - Parameter coords: `[b, n, 37, 3]`, Angstrom, NaN where unconditioned.
    /// - Returns: `[b, n, 37 * 4]` — flattened coordinates then the presence mask.
    public static func conditionalAtom37Coords(
        _ coords: MLXArray, relative: Bool = false
    ) -> MLXArray {
        let finite = MLX.any(MLX.isFinite(coords), axis: -1)          // [b, n, 37]
        let maskF = finite.asType(.float32)
        var nm = MLX.where(MLX.isNaN(coords), MLXArray(Float(0)), coords) / 10
        nm = nm * maskF.expandedDimensions(axis: -1)

        if relative {
            // Slot 1 is CA; offsets are measured from it.
            let ca = nm[0..., 0..., 1, 0...].expandedDimensions(axis: 2)
            nm = (nm - ca) * maskF.expandedDimensions(axis: -1)
        }

        let (b, n) = (coords.dim(0), coords.dim(1))
        let flat = nm.reshaped(b, n, Vocabulary.slotCount * 3)
        return MLX.concatenated([flat, maskF], axis: -1)
    }

    /// Hotspot annotation — `HotspotSeqFeat`.
    ///
    /// `[b, n, 37]` of hotspot flags, gated by which atom37 slots actually hold an
    /// atom, so a hotspot cannot be declared on a slot the residue does not have.
    public static func hotspot(_ hot: MLXArray, atom37Mask: MLXArray) -> MLXArray {
        hot.asType(.float32) * atom37Mask.asType(.float32)
    }

    /// Sequence probabilities — `SequenceProbabilitiesSeqFeat`.
    ///
    /// `[b, n, 2 * K]`: the probabilities, then a mask of which were explicitly
    /// specified. The mask is the point — it makes an explicit probability of zero
    /// ("definitely not this residue") distinguishable from an unconstrained
    /// position, which the probability value alone cannot express. Unspecified
    /// entries arrive as NaN.
    ///
    /// - Parameter channels: `K` arrays of `[b, n]`, in registry order.
    public static func sequenceProbabilities(_ channels: [MLXArray]) -> MLXArray {
        guard !channels.isEmpty else { return MLXArray([Float]()) }
        let stacked = MLX.stacked(channels, axis: -1)                 // [b, n, K]
        let specified = MLX.logicalNot(MLX.isNaN(stacked))
        let values = MLX.where(specified, stacked, MLXArray(Float(0)))
        return MLX.concatenated([values, specified.asType(.float32)], axis: -1)
    }

    /// Backbone linkage direction — `CanonicalPolymerOrderingPairFeat`.
    ///
    /// A pass-through of the `[b, n, n]` array the data pipeline builds: `+1` at
    /// `[downstream, upstream]`, `-1` at the transpose, `0` elsewhere. Signed, so it
    /// encodes chain direction rather than mere adjacency.
    public static func canonicalPolymerOrdering(_ ordering: MLXArray) -> MLXArray {
        ordering.asType(.float32).expandedDimensions(axis: -1)
    }

    /// A zero block standing in for a feature that is inert in this regime.
    ///
    /// Explicit rather than implicit: the concatenation must still be the right
    /// width, and naming the gap makes it obvious which features are unimplemented
    /// rather than merely absent.
    public static func inert(_ shape: [Int]) -> MLXArray { MLX.zeros(shape) }
}
