import Foundation
import MLX

/// SwiGLU feed-forward — `nn/modules/seq_transition_af3.py:Transition`.
///
/// `linear_out(SwiGLU(swish_linear(x))) * mask`, both projections bias-free.
/// The inner projection is `dim -> 2 * dim * expansion`, halved again by the gate,
/// so at 768 with expansion 4 it is 768 -> 6144 -> 3072 -> 768.
public struct Transition {
    let inW: MLXArray
    let outW: MLXArray

    public init(_ w: Weights) throws {
        inW = try w("swish_linear.0.weight")
        outW = try w("linear_out.weight")
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = swiGLU(linear(x, inW))
        return linear(h, outW) * mask.expandedDimensions(axis: -1)
    }
}

/// `TransitionADALN`: adaptive norm in, SwiGLU MLP, adaptive scale out.
public struct TransitionADALN {
    let adaln: AdaptiveLayerNorm
    let transition: Transition
    let scaleOutput: AdaptiveOutputScale

    public init(_ w: Weights, kind: NormKind) throws {
        adaln = try AdaptiveLayerNorm(w.scoped("adaln"), kind: kind)
        transition = try Transition(w.scoped("transition"))
        scaleOutput = try AdaptiveOutputScale(w.scoped("scale_output"))
    }

    public func callAsFunction(
        _ x: MLXArray, cond: MLXArray, mask: MLXArray
    ) -> MLXArray {
        var y = adaln(x, cond: cond, mask: mask)
        y = transition(y, mask: mask)
        y = scaleOutput(y, cond: cond, mask: mask)
        return y * mask.expandedDimensions(axis: -1)
    }
}

/// One trunk layer — `nn/modules/attn_n_transition.py:MultiheadAttnAndTransition`.
///
/// The released checkpoint sets `parallel_mha_transition: False` and keeps both
/// residual connections, so attention and the MLP run in sequence, each adding its
/// input back. The mask is re-applied after every sub-step, which upstream does
/// defensively and which is reproduced here rather than optimised away: it is what
/// keeps padded positions at exactly zero instead of at whatever a bias leaked in.
public struct TrunkLayer {
    let mhba: PairBiasAttentionADALN
    let transition: TransitionADALN
    let residualMHA: Bool
    let residualTransition: Bool

    public init(
        _ w: Weights, heads: Int, dimHead: Int, kind: NormKind,
        residualMHA: Bool = true, residualTransition: Bool = true
    ) throws {
        mhba = try PairBiasAttentionADALN(w.scoped("mhba"), heads: heads, dimHead: dimHead, kind: kind)
        transition = try TransitionADALN(w.scoped("transition"), kind: kind)
        self.residualMHA = residualMHA
        self.residualTransition = residualTransition
    }

    public func callAsFunction(
        _ x: MLXArray, pairRep: MLXArray?, cond: MLXArray, mask: MLXArray
    ) -> MLXArray {
        let m = mask.expandedDimensions(axis: -1)
        var h = x * m

        var attn = mhba(h, pairRep: pairRep, cond: cond, mask: mask)
        if residualMHA { attn = attn + h }
        h = attn * m

        var tr = transition(h, cond: cond, mask: mask)
        if residualTransition { tr = tr + h }
        h = tr * m

        return h * m
    }
}
