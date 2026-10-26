import Foundation
import MLX

/// Pair-biased, gated multi-head self-attention — `nn/modules/pair_bias_attn.py`.
///
/// The one attention variant in this network. It is ordinary scaled dot-product
/// attention plus three things:
///   - an **additive pair bias** `[b, h, n, n]` projected from the pair representation
///   - **QK normalisation** over the *full* inner dimension (768), not per head
///   - a **sigmoid gate** on the attended values before the output projection
///
/// The explicit einsum form below is upstream's own CPU reference
/// (`pair_bias_attn.py:173`), which is what runs on a machine without CUDA. Porting
/// that rather than the SDPA branch keeps the arithmetic comparable term by term.
public struct PairBiasAttention {
    /// Upstream uses -1e4, **not** `-infinity` (`pair_bias_attn.py:40`). With -inf a
    /// fully masked row softmaxes to NaN; -1e4 keeps it finite, and the row fixup
    /// below then zeroes the bias for rows that are masked everywhere.
    public static let negInf: Float = -1e4

    let heads: Int, dimHead: Int
    let nodeNorm: MLXArray
    let qkvW: MLXArray, qkvB: MLXArray
    let gW: MLXArray, gB: MLXArray
    let qNorm: MLXArray, kNorm: MLXArray
    let outW: MLXArray, outB: MLXArray
    let pairNorm: MLXArray, biasW: MLXArray
    let eps: Float

    public init(_ w: Weights, heads: Int, dimHead: Int, eps: Float = 1e-5) throws {
        self.heads = heads
        self.dimHead = dimHead
        self.eps = eps
        nodeNorm = try w("node_norm.weight")
        qkvW = try w("to_qkv.weight")
        qkvB = try w("to_qkv.bias")
        gW = try w("to_g.weight")
        gB = try w("to_g.bias")
        qNorm = try w("q_layer_norm.weight")
        kNorm = try w("k_layer_norm.weight")
        outW = try w("to_out_node.weight")
        outB = try w("to_out_node.bias")
        pairNorm = try w("pair_norm.weight")
        biasW = try w("to_bias.weight")   // Linear(pair_dim -> heads, bias=False)
    }

    /// `rearrange(t, "b n (h d) -> b h n d")`
    private func splitHeads(_ t: MLXArray) -> MLXArray {
        let (b, n) = (t.dim(0), t.dim(1))
        return t.reshaped(b, n, heads, dimHead).transposed(0, 2, 1, 3)
    }

    /// - Parameters:
    ///   - nodeFeats: `[b, n, dim]`
    ///   - pairFeats: `[b, n, n, pairDim]`
    ///   - pairMask: `[b, n, n]`, float — 1 where attention is allowed
    public func callAsFunction(
        _ nodeFeats: MLXArray, pairFeats: MLXArray, pairMask: MLXArray
    ) -> MLXArray {
        let node = rmsNorm(nodeFeats, nodeNorm, eps: eps)
        let inner = heads * dimHead

        let qkv = linear(node, qkvW, qkvB)
        // chunk(3, dim=-1)
        var q = qkv[.ellipsis, 0 ..< inner]
        var k = qkv[.ellipsis, inner ..< (2 * inner)]
        let v = qkv[.ellipsis, (2 * inner) ..< (3 * inner)]

        // QK-LayerNorm is applied to the whole [b, n, 768] projection BEFORE the
        // head split, so it normalises across all heads together. Normalising per
        // head instead changes the result and is an easy transcription error.
        q = rmsNorm(q, qNorm, eps: eps)
        k = rmsNorm(k, kNorm, eps: eps)
        let g = linear(node, gW, gB)

        let qh = splitHeads(q), kh = splitHeads(k), vh = splitHeads(v), gh = splitHeads(g)

        // Pair bias: [b, n, n, pairDim] -> [b, n, n, h] -> [b, h, n, n]
        let pair = rmsNorm(pairFeats, pairNorm, eps: eps)
        var attnBias = linear(pair, biasW).transposed(0, 3, 1, 2)

        // mask [b, n, n] -> [b, 1, n, n]
        let maskExpanded = pairMask.expandedDimensions(axis: 1)
        let allowed = maskExpanded .> 0
        attnBias = MLX.where(allowed, attnBias, MLXArray(Self.negInf))

        // A row masked at every column would softmax over all -1e4 and contribute
        // garbage; upstream zeroes the bias for those rows instead.
        let rowHasAny = MLX.sum(maskExpanded, axis: -1, keepDims: true) .> 0
        attnBias = MLX.where(rowHasAny, attnBias, MLXArray(Float(0)))

        // einsum("b h i d, b h j d -> b h i j")
        let scale = MLXArray(pow(Float(dimHead), -0.5))
        let sim = MLX.matmul(qh, kh.transposed(0, 1, 3, 2)) * scale + attnBias
        let attn = MLX.softMax(sim, axis: -1)

        // einsum("b h i j, b h j d -> b h i d")
        let o = MLX.matmul(attn, vh)
        let gated = MLX.sigmoid(gh) * o

        // rearrange("b h n d -> b n (h d)")
        let (b, n) = (gated.dim(0), gated.dim(2))
        let merged = gated.transposed(0, 2, 1, 3).reshaped(b, n, inner)
        return linear(merged, outW, outB)
    }
}

/// `MultiHeadBiasedAttentionADALN_MM`: adaptive norm in, attention, adaptive scale out.
public struct PairBiasAttentionADALN {
    let adaln: AdaptiveLayerNorm
    let mha: PairBiasAttention
    let scaleOutput: AdaptiveOutputScale

    public init(_ w: Weights, heads: Int, dimHead: Int) throws {
        adaln = try AdaptiveLayerNorm(w.scoped("adaln"))
        mha = try PairBiasAttention(w.scoped("mha"), heads: heads, dimHead: dimHead)
        scaleOutput = try AdaptiveOutputScale(w.scoped("scale_output"))
    }

    /// - Parameter mask: `[b, n]` float. The pair mask is its outer product, formed
    ///   here exactly as upstream does: `mask[:, :, None] * mask[:, None, :]`.
    public func callAsFunction(
        _ x: MLXArray, pairRep: MLXArray, cond: MLXArray, mask: MLXArray
    ) -> MLXArray {
        let pairMask = mask.expandedDimensions(axis: 2) * mask.expandedDimensions(axis: 1)
        var y = adaln(x, cond: cond, mask: mask)
        y = mha(y, pairFeats: pairRep, pairMask: pairMask)
        y = scaleOutput(y, cond: cond, mask: mask)
        return y * mask.expandedDimensions(axis: -1)
    }
}
