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
    let zW: MLXArray
    let eps: Float

    /// `linear_a_p`, `linear_a_g`, `linear_b_p`, `linear_b_g` and `linear_g` stacked
    /// into one `[256+256+256+256+384, 384]` matrix.
    ///
    /// All five read the SAME `zn`, which is `[b, n, n, 384]` — 0.58 GB at n=638.
    /// Run separately they stream it five times; run as one matmul they stream it
    /// once and the GPU gets a wider, better-shaped problem. The result is sliced
    /// back apart, so this is exact rather than an approximation: concatenating
    /// along the output dimension and slicing is the definition of what the five
    /// calls computed.
    let fusedW: MLXArray
    static let aSize = 256, gateSize = 384

    /// Gate chain: (fused, m) → (a, b).
    ///
    /// Without compile: 4 slices + 2 sigmoid + 4 multiply = 10 dispatches and 9
    /// temporary [b, n, n, 256] allocations (157 MB each at n=638). With compile,
    /// MLX fuses into one Metal kernel that reads `fused` and `m` once and writes
    /// `a` and `b` without any intermediate allocation.
    private static let compiledGate: @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray) =
        compile { (fused: MLXArray, m: MLXArray) -> (MLXArray, MLXArray) in
            let c = TriangleMultiplication.aSize
            let ap = fused[.ellipsis, 0 ..< c]
            let ag = fused[.ellipsis, c ..< (2 * c)]
            let bp = fused[.ellipsis, (2 * c) ..< (3 * c)]
            let bg = fused[.ellipsis, (3 * c) ..< (4 * c)]
            return (ap * MLX.sigmoid(ag) * m, bp * MLX.sigmoid(bg) * m)
        }

    /// Output gate: (x, gl) → x * sigmoid(gl) — compiled to avoid a temporary.
    private static let compiledOutputGate: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile { (x: MLXArray, gl: MLXArray) -> MLXArray in
            x * MLX.sigmoid(gl)
        }

    public init(_ w: Weights, direction: Direction, eps: Float = 1e-5) throws {
        self.direction = direction
        self.eps = eps
        normInW = try w("layer_norm_in.weight")
        normInB = w.optional("layer_norm_in.bias")
        normOutW = try w("layer_norm_out.weight")
        normOutB = w.optional("layer_norm_out.bias")
        // Pre-transpose at load so the forward matmul uses the B-normal GEMM path,
        // which is faster than the B-transposed path for these wide pair-track shapes.
        zW = MLX.contiguous((try w("linear_z.weight")).transposed())
        // Stacked once at load, then pre-transposed for the same GEMM reason.
        fusedW = MLX.contiguous(MLX.concatenated([
            try w("linear_a_p.weight"), try w("linear_a_g.weight"),
            try w("linear_b_p.weight"), try w("linear_b_g.weight"),
            try w("linear_g.weight"),
        ], axis: 0).transposed())
    }

    /// `permute_final_dims` on a 4-D tensor, as the two directions use it.
    /// Outgoing: `a[b,i,k,c] -> [b,c,i,k]`, `b[b,j,k,c] -> [b,c,k,j]`.
    /// Incoming: `a[b,k,i,c] -> [b,c,i,k]`, `b[b,k,j,c] -> [b,c,k,j]`.
    private func combine(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let (am, bm): (MLXArray, MLXArray) = Profile.section("trimult.transposeIn", {
            switch direction {
            case .outgoing:
                return (a.transposed(0, 3, 1, 2), b.transposed(0, 3, 2, 1))
            case .incoming:
                return (a.transposed(0, 3, 2, 1), b.transposed(0, 3, 1, 2))
            }
        }, force: { [$0.0, $0.1] })
        // [b, c, i, j] -> [b, i, j, c]
        let prod = Profile.section("trimult.matmul") { MLX.matmul(am, bm) }
        return Profile.section("trimult.transposeOut") { prod.transposed(0, 2, 3, 1) }
    }

    /// - Parameters:
    ///   - z: `[b, n, n, c_z]`
    ///   - pairMask: `[b, n, n]`
    public func callAsFunction(_ z: MLXArray, pairMask: MLXArray) -> MLXArray {
        let m = pairMask.expandedDimensions(axis: -1)
        let zn = layerNorm(z, normInW, normInB, eps: eps)

        // One matmul, then slice. `linear` is x @ Wᵀ, so stacking along the output
        // dimension puts each projection in its own contiguous band of the result.
        let fused = Profile.section("trimult.projections") { linearT(zn, fusedW) }
        let c = Self.aSize
        let gl = fused[.ellipsis, (4 * c) ..< (4 * c + Self.gateSize)]
        let (a, b) = Self.compiledGate(fused, m)

        var x = combine(a, b)
        return Profile.section("trimult.output") {
            x = layerNorm(x, normOutW, normOutB, eps: eps)
            x = linearT(x, zW)
            return Self.compiledOutputGate(x, gl)
        }
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
        // Pre-transposed for the faster B-normal GEMM path.
        w1 = MLX.contiguous((try w("linear_1.weight")).transposed())
        b1 = w.optional("linear_1.bias")
        w2 = MLX.contiguous((try w("linear_2.weight")).transposed())
        b2 = w.optional("linear_2.bias")
        self.eps = eps
    }

    public func callAsFunction(_ z: MLXArray, pairMask: MLXArray) -> MLXArray {
        let m = pairMask.expandedDimensions(axis: -1)
        var y = layerNorm(z, normW, normB, eps: eps)
        y = MLX.maximum(linearT(y, w1, b1), 0)      // ReLU
        return linearT(y, w2, b2) * m
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

    /// Masked residual add: (z, update, pm) → (z + update) * pm.
    ///
    /// Without compile: two dispatches (add + multiply) over a 0.58 GB tensor.
    /// With compile: one kernel, called three times per pair update.
    private static let compiledResidual: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
        compile { (z: MLXArray, update: MLXArray, pm: MLXArray) -> MLXArray in
            (z + update) * pm
        }

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

        z = Self.compiledResidual(z, triOut(z, pairMask: pairMask), pm)
        z = Self.compiledResidual(z, triIn(z, pairMask: pairMask), pm)
        z = Profile.section("pair.transition") {
            Self.compiledResidual(z, transition(z, pairMask: pairMask), pm)
        }
        return z
    }
}
