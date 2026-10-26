import Foundation
import MLX

/// Conditioned normalization — `nn/modules/adaptive_ln_scale.py:AdaptiveLayerNorm`.
///
/// ```python
/// normed = self.norm(x)                      # RMSNorm, elementwise_affine=False
/// normed_cond = self.norm_cond(cond)         # RMSNorm, affine
/// gamma = sigmoid(Linear(normed_cond))       # to_gamma = Sequential(Linear, Sigmoid)
/// beta  = Linear(normed_cond)                # bias=False
/// return (normed * gamma + beta) * mask[..., None]
/// ```
///
/// Upstream also broadcasts a `[b, dim_cond]` conditioning vector across the spatial
/// axes when one is given. The trunk's layers always pass `cond` as
/// `[b, n, dim_cond]` and never reach that branch — but `PairReprBuilder` builds its
/// conditioning with `compact: true`, which yields exactly `[b, dim_cond]` against a
/// 4-D pair representation. So the branch is live, and both shapes are handled.
public struct AdaptiveLayerNorm {
    /// `elementwise_affine: False` — no weight, hence a bare normalisation.
    let norm: Norm
    let normCond: Norm
    let gammaW: MLXArray, gammaB: MLXArray
    let betaW: MLXArray

    public init(_ w: Weights, kind: NormKind, eps: Float = 1e-5) throws {
        norm = Norm(w.scoped("norm"), kind: kind, eps: eps)
        normCond = Norm(w.scoped("norm_cond"), kind: kind, eps: eps)
        gammaW = try w("to_gamma.0.weight")
        gammaB = try w("to_gamma.0.bias")
        betaW = try w("to_beta.weight")
    }

    /// - Parameters:
    ///   - x: `[b, n, dim]`
    ///   - cond: `[b, n, dimCond]`
    ///   - mask: `[b, n]`, float
    public func callAsFunction(_ x: MLXArray, cond: MLXArray, mask: MLXArray) -> MLXArray {
        let normed = norm(x)
        let normedCond = normCond(cond)
        var gamma = MLX.sigmoid(linear(normedCond, gammaW, gammaB))
        var beta = linear(normedCond, betaW)

        // Per-example conditioning: reshape [b, dimCond] to [b, 1, ..., 1, dim] so it
        // broadcasts across every spatial axis of x.
        if cond.ndim == 2 && x.ndim > 2 {
            let shape = [cond.dim(0)] + Array(repeating: 1, count: x.ndim - 2) + [x.dim(-1)]
            gamma = gamma.reshaped(shape)
            beta = beta.reshaped(shape)
        }
        return (normed * gamma + beta) * mask.expandedDimensions(axis: -1)
    }
}

/// Adaptive output scaling — `adaptive_ln_scale.py:AdaptiveOutputScale`.
///
/// `return x * sigmoid(Linear(cond)) * mask[..., None]`
///
/// The gamma Linear is zero-initialised with bias `-2.0`, so at initialisation the
/// gate is `sigmoid(-2) ~= 0.119` and each block starts as a near-identity residual.
/// That matters for reading trained values, not for the forward pass.
public struct AdaptiveOutputScale {
    let gammaW: MLXArray, gammaB: MLXArray

    public init(_ w: Weights) throws {
        gammaW = try w("to_adaln_zero_gamma.0.weight")
        gammaB = try w("to_adaln_zero_gamma.0.bias")
    }

    public func callAsFunction(_ x: MLXArray, cond: MLXArray, mask: MLXArray) -> MLXArray {
        let gamma = MLX.sigmoid(linear(cond, gammaW, gammaB))
        return x * gamma * mask.expandedDimensions(axis: -1)
    }
}
