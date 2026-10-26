import Foundation
import MLX

/// The flow trunk — `nn/local_latents_transformer.py:LocalLatentsTransformer`.
///
/// 14 attention+MLP layers over the single representation, with a pair update after
/// every second layer, wrapped in a recycling loop. The featurization that builds
/// `sInit`, `zInit` and `cond` is not part of this type; it takes them as given, so
/// the network can be exercised and checked long before the featurizer exists.
public struct Trunk {
    public static let layerCount = 14

    let layers: [TrunkLayer]
    /// 13 slots, `nil` at odd indices. Not 7 evenly spaced updates: upstream keeps a
    /// `ModuleList` of 13 with holes, and runs `pair_layers[i]` only when `i` is below
    /// the last layer and the slot is populated — so there is no pair update after
    /// layer 13, and none after any odd layer.
    let pairUpdates: [PairReprUpdate?]

    let sCycleNorm: MLXArray, sCycleProj: MLXArray
    let zCycleNorm: MLXArray, zCycleProj: MLXArray

    let localLatentsHead: Head
    let caHead: Head
    let sequenceHead: Head?
    let insertionQ: MLXArray, insertionK: MLXArray
    let insertionScale: Float

    let eps: Float

    /// `Sequential(make_norm(dim), Linear(dim, out, bias=False))` — the shape every
    /// readout head takes, stored as `.0.weight` (norm) and `.1.weight` (projection).
    public struct Head {
        let normW: MLXArray
        let projW: MLXArray
        let eps: Float

        init(_ w: Weights, eps: Float) throws {
            normW = try w("0.weight")
            projW = try w("1.weight")
            self.eps = eps
        }

        func callAsFunction(_ s: MLXArray) -> MLXArray {
            linear(rmsNorm(s, normW, eps: eps), projW)
        }
    }

    public struct Output {
        public let bbCA: MLXArray            // [b, n, 3]
        public let localLatents: MLXArray    // [b, n, 32]
        public let sequenceLogits: MLXArray? // [b, n, 147]
        public let insertionLogits: MLXArray // [b, n, n]
        public let s: MLXArray
        public let z: MLXArray
    }

    public init(
        _ w: Weights,
        heads: Int = 12,
        dimHead: Int = 64,
        tokenDim: Int = 768,
        eps: Float = 1e-5
    ) throws {
        self.eps = eps
        let root = w.scoped("nn")

        layers = try (0 ..< Self.layerCount).map {
            try TrunkLayer(root.scoped("transformer_layers", $0),
                           heads: heads, dimHead: dimHead, kind: .rms)
        }
        pairUpdates = try (0 ..< (Self.layerCount - 1)).map { i -> PairReprUpdate? in
            let scoped = root.scoped("pair_update_layers", i)
            // An absent slot is upstream's `None`, not a missing weight.
            guard scoped.has("linear_x.weight") else { return nil }
            return try PairReprUpdate(scoped)
        }

        sCycleNorm = try root("s_cycle_norm.weight")
        sCycleProj = try root("s_cycle_proj.weight")
        zCycleNorm = try root("z_cycle_norm.weight")
        zCycleProj = try root("z_cycle_proj.weight")

        localLatentsHead = try Head(root.scoped("local_latents_linear"), eps: eps)
        caHead = try Head(root.scoped("ca_linear"), eps: eps)
        sequenceHead = root.has("sequence_readout.1.weight")
            ? try Head(root.scoped("sequence_readout"), eps: eps) : nil
        insertionQ = try root("insertion_q_proj.weight")
        insertionK = try root("insertion_k_proj.weight")
        insertionScale = pow(Float(tokenDim), -0.5)
    }

    /// One trunk pass: 14 layers, pair update after every populated even slot.
    public func applyTrunk(
        s: MLXArray, z: MLXArray, cond: MLXArray, mask: MLXArray
    ) -> (s: MLXArray, z: MLXArray) {
        var s = s, z = z
        for i in 0 ..< Self.layerCount {
            s = layers[i](s, pairRep: z, cond: cond, mask: mask)
            if i < Self.layerCount - 1, let pu = pairUpdates[i] {
                z = pu(s, pairRep: z, mask: mask)
            }
        }
        return (s, z)
    }

    /// The recycling loop.
    ///
    /// `s` and `z` start at zero and are fed back through zero-initialised,
    /// bias-free projections. **No coordinate recycling** — unlike AlphaFold, the
    /// noised structure `x_t` is unchanged across cycles; only the two
    /// representations carry forward.
    ///
    /// `nRecycle` is floored at 1 (`local_latents_transformer.py:386`), so a depth of
    /// 0 still runs the trunk once. The per-cycle rebuild of `cond` from
    /// `recycle_remaining` is deliberately not reproduced: under the released
    /// `recycle.grad: "last"` that feature one-hots to a constant `[1.0]`, so it
    /// cannot vary. If a checkpoint ever ships a wider setting, this is where it goes.
    public func runTrunk(
        sInit: MLXArray, zInit: MLXArray, cond: MLXArray, mask: MLXArray, nRecycle: Int = 1
    ) -> (s: MLXArray, z: MLXArray) {
        var s = MLX.zeros(like: sInit)
        var z = MLX.zeros(like: zInit)
        for _ in 0 ..< max(1, nRecycle) {
            let zIn = zInit + linear(rmsNorm(z, zCycleNorm, eps: eps), zCycleProj)
            let sIn = sInit + linear(rmsNorm(s, sCycleNorm, eps: eps), sCycleProj)
            (s, z) = applyTrunk(s: sIn, z: zIn, cond: cond, mask: mask)
        }
        return (s, z)
    }

    /// Trunk plus the four readout heads.
    ///
    /// - Parameters:
    ///   - emitBBCA / emitLocalLatents: `[b, n]` emit masks. Upstream derives these
    ///     from the batch; they are passed in because that derivation belongs to the
    ///     featurizer, not the network.
    public func callAsFunction(
        sInit: MLXArray, zInit: MLXArray, cond: MLXArray, mask: MLXArray,
        emitBBCA: MLXArray, emitLocalLatents: MLXArray, nRecycle: Int = 1
    ) -> Output {
        let (s, z) = runTrunk(sInit: sInit, zInit: zInit, cond: cond,
                              mask: mask, nRecycle: nRecycle)

        let latents = localLatentsHead(s) * emitLocalLatents.expandedDimensions(axis: -1)
        let ca = caHead(s) * emitBBCA.expandedDimensions(axis: -1)
        let seq = sequenceHead.map { $0(s) * mask.expandedDimensions(axis: -1) }

        // Bilinear outer product over all token pairs, scaled by tokenDim^-0.5.
        let q = linear(s, insertionQ)
        let k = linear(s, insertionK)
        let insertion = MLX.matmul(q, k.transposed(0, 2, 1)) * insertionScale

        return Output(bbCA: ca, localLatents: latents, sequenceLogits: seq,
                      insertionLogits: insertion, s: s, z: z)
    }
}
