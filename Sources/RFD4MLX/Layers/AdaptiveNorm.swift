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
/// Upstream also carries a branch that broadcasts a `[b, dim_cond]` conditioning
/// vector across spatial axes. The trunk always passes `cond` as `[b, n, dim_cond]`,
/// so that branch is dead here and is deliberately not ported — porting it would
/// mean reproducing a reshape no input reaches.
public struct AdaptiveLayerNorm {
    let condWeight: MLXArray      // norm_cond.weight
    let gammaW: MLXArray, gammaB: MLXArray
    let betaW: MLXArray
    let eps: Float

    public init(_ w: Weights, eps: Float = 1e-5) throws {
        condWeight = try w("norm_cond.weight")
        gammaW = try w("to_gamma.0.weight")
        gammaB = try w("to_gamma.0.bias")
        betaW = try w("to_beta.weight")
        self.eps = eps
    }

    /// - Parameters:
    ///   - x: `[b, n, dim]`
    ///   - cond: `[b, n, dimCond]`
    ///   - mask: `[b, n]`, float
    public func callAsFunction(_ x: MLXArray, cond: MLXArray, mask: MLXArray) -> MLXArray {
        // norm has no weight: elementwise_affine=False upstream.
        let normed = rmsNorm(x, nil, eps: eps)
        let normedCond = rmsNorm(cond, condWeight, eps: eps)
        let gamma = MLX.sigmoid(linear(normedCond, gammaW, gammaB))
        let beta = linear(normedCond, betaW)
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
