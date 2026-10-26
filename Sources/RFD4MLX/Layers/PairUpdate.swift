import Foundation
import MLX

/// Triangle multiplicative update — `openfold_modules/triangular_multiplicative_update.py`.
///
/// The only O(n³) operation in the network, and the one that sets the device budget:
/// it materialises a `[b, c_hidden, n, n]` intermediate, 208 MB at 638 tokens.
///
/// Upstream has a fused cuEquivariance path, but it is guarded on `z.is_cuda` and
/// falls back to the plain torch code below on anything else — so the fallback *is*
/// the reference, not an approximation of it.
///
/// Upstream also fuses the four projections into one grouped GEMM under no-grad.
/// That is an allocation optimisation: each output element is the same dot product
/// either way, so the separate projections here are bit-identical, not merely close.
public struct TriangleMultiplication {
    public enum Direction { case outgoing, incoming }

    let direction: Direction
    let normInW: MLXArray, normInB: MLXArray?
    let normOutW: MLXArray, normOutB: MLXArray?
    let apW: MLXArray, agW: MLXArray
    let bpW: MLXArray, bgW: MLXArray
    let zW: MLXArray, gW: MLXArray
    let eps: Float

    public init(_ w: Weights, direction: Direction, eps: Float = 1e-5) throws {
        self.direction = direction
        self.eps = eps
        normInW = try w("layer_norm_in.weight")
        normInB = w.optional("layer_norm_in.bias")
        normOutW = try w("layer_norm_out.weight")
        normOutB = w.optional("layer_norm_out.bias")
        apW = try w("linear_a_p.weight")
        agW = try w("linear_a_g.weight")
        bpW = try w("linear_b_p.weight")
        bgW = try w("linear_b_g.weight")
        zW = try w("linear_z.weight")
        gW = try w("linear_g.weight")
    }

    /// `permute_final_dims` on a 4-D tensor, as the two directions use it.
    /// Outgoing: `a[b,i,k,c] -> [b,c,i,k]`, `b[b,j,k,c] -> [b,c,k,j]`.
    /// Incoming: `a[b,k,i,c] -> [b,c,i,k]`, `b[b,k,j,c] -> [b,c,k,j]`.
    private func combine(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let (am, bm): (MLXArray, MLXArray)
        switch direction {
        case .outgoing:
            am = a.transposed(0, 3, 1, 2)
            bm = b.transposed(0, 3, 2, 1)
        case .incoming:
            am = a.transposed(0, 3, 2, 1)
            bm = b.transposed(0, 3, 1, 2)
        }
        // [b, c, i, j] -> [b, i, j, c]
        return MLX.matmul(am, bm).transposed(0, 2, 3, 1)
    }

    /// - Parameters:
    ///   - z: `[b, n, n, c_z]`
    ///   - pairMask: `[b, n, n]`
    public func callAsFunction(_ z: MLXArray, pairMask: MLXArray) -> MLXArray {
        let m = pairMask.expandedDimensions(axis: -1)
        let zn = layerNorm(z, normInW, normInB, eps: eps)

        let a = linear(zn, apW) * MLX.sigmoid(linear(zn, agW)) * m
        let b = linear(zn, bpW) * MLX.sigmoid(linear(zn, bgW)) * m

        var x = combine(a, b)
        x = layerNorm(x, normOutW, normOutB, eps: eps)
        x = linear(x, zW)
        let g = MLX.sigmoid(linear(zn, gW))
        return x * g
    }
}

/// Pair transition — `openfold_modules/pair_transition.py`. LayerNorm, expand, ReLU,
/// project back, mask. Note this is ReLU, not the SwiGLU the sequence track uses.
public struct PairTransition {
    let normW: MLXArray, normB: MLXArray?
    let w1: MLXArray, b1: MLXArray?
    let w2: MLXArray, b2: MLXArray?
    let eps: Float

    public init(_ w: Weights, eps: Float = 1e-5) throws {
        normW = try w("layer_norm.weight")
        normB = w.optional("layer_norm.bias")
        w1 = try w("linear_1.weight")
        b1 = w.optional("linear_1.bias")
        w2 = try w("linear_2.weight")
        b2 = w.optional("linear_2.bias")
        self.eps = eps
    }

    public func callAsFunction(_ z: MLXArray, pairMask: MLXArray) -> MLXArray {
        let m = pairMask.expandedDimensions(axis: -1)
        var y = layerNorm(z, normW, normB, eps: eps)
        y = MLX.maximum(linear(y, w1, b1), 0)      // ReLU
        return linear(y, w2, b2) * m
    }
}

/// `nn/modules/pair_update.py:PairReprUpdate`.
///
/// Writes the single representation into the pair track, then runs both triangle
/// multiplications and the pair transition, each as a masked residual.
///
/// The released checkpoint sets `pair_write: "hadamard"`, **not** the `"sum"`
/// default in the constructor signature — the single-to-pair write multiplies the
/// two projections rather than adding them. Reading the default off the source
/// would silently produce a different network.
public struct PairReprUpdate {
    public enum Write { case sum, hadamard }

    let normInW: MLXArray, normInB: MLXArray?
    let linearXW: MLXArray
    let triOut: TriangleMultiplication
    let triIn: TriangleMultiplication
    let transition: PairTransition
    let write: Write
    let eps: Float

    public init(_ w: Weights, write: Write = .hadamard, eps: Float = 1e-5) throws {
        normInW = try w("layer_norm_in.weight")
        normInB = w.optional("layer_norm_in.bias")
        linearXW = try w("linear_x.weight")
        triOut = try TriangleMultiplication(w.scoped("tri_mult_out"), direction: .outgoing)
        triIn = try TriangleMultiplication(w.scoped("tri_mult_in"), direction: .incoming)
        transition = try PairTransition(w.scoped("transition_out"))
        self.write = write
        self.eps = eps
    }

    /// - Parameters:
    ///   - x: `[b, n, tokenDim]` single representation
    ///   - pairRep: `[b, n, n, pairDim]`
    ///   - mask: `[b, n]`
    public func callAsFunction(
        _ x: MLXArray, pairRep: MLXArray, mask: MLXArray
    ) -> MLXArray {
        // Upstream builds this as mask[:, None, :] * mask[:, :, None] — the
        // transpose of the attention block's order. Symmetric, so it does not
        // matter, but kept in the same orientation as the source it came from.
        let pairMask = mask.expandedDimensions(axis: 1) * mask.expandedDimensions(axis: 2)
        let pm = pairMask.expandedDimensions(axis: -1)

        let xm = x * mask.expandedDimensions(axis: -1)
        let proj = linear(layerNorm(xm, normInW, normInB, eps: eps), linearXW)
        let half = proj.dim(-1) / 2
        let pi = proj[.ellipsis, 0 ..< half]
        let pj = proj[.ellipsis, half ..< (2 * half)]

        // pi broadcasts along i (column write), pj along j (row write).
        let piB = pi.expandedDimensions(axis: 1)
        let pjB = pj.expandedDimensions(axis: 2)

        var z = write == .sum ? pairRep + piB + pjB : pairRep + piB * pjB
        z = z * pm

        z = (z + triOut(z, pairMask: pairMask)) * pm
        z = (z + triIn(z, pairMask: pairMask)) * pm
        z = (z + transition(z, pairMask: pairMask)) * pm
        return z
    }
}
