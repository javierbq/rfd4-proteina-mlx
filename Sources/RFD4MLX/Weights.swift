import Foundation
import MLX

/// A view over the checkpoint's flat tensor dictionary, scoped to one module.
///
/// The checkpoint is a plain safetensors file whose keys are the PyTorch module
/// paths (`nn.transformer_layers.0.mhba.mha.to_qkv.weight`). Rather than mirror
/// PyTorch's module tree in Swift and fight two naming conventions, each layer
/// reads the keys it owns from a prefixed view of that dictionary.
///
/// A missing key is an error, never a zero: a silently absent weight is the exact
/// failure a parity harness is built to catch, and it would otherwise show up as a
/// plausible-looking wrong structure many layers later.
public struct Weights {
    public let arrays: [String: MLXArray]
    public let prefix: String

    public init(arrays: [String: MLXArray], prefix: String = "") {
        self.arrays = arrays
        self.prefix = prefix
    }

    /// Load a safetensors checkpoint.
    /// The precision the network's weights are held and computed in.
    ///
    /// `bfloat16` keeps fp32's exponent range and drops mantissa bits, which is the
    /// right trade for this network: the pair track moves hundreds of megabytes per
    /// layer, so halving every tensor halves the traffic, and MLX accumulates matmuls
    /// in fp32 regardless. It is **not** exact, so the parity suite alone cannot
    /// clear it — the matched-noise harness is what measures the cost, by driving
    /// both precisions down upstream's own trajectory so the difference that remains
    /// is precision rather than sampling.
    public enum Precision: String, Sendable {
        case float32, bfloat16

        /// `RFD4_DTYPE=bf16` switches a run over without a rebuild.
        public static var fromEnvironment: Precision {
            switch ProcessInfo.processInfo.environment["RFD4_DTYPE"]?.lowercased() {
            case "bf16", "bfloat16": return .bfloat16
            default: return .float32
            }
        }

        var dtype: DType { self == .bfloat16 ? .bfloat16 : .float32 }
    }

    public static func load(_ url: URL, precision: Precision = .fromEnvironment) throws
        -> Weights {
        let raw = try MLX.loadArrays(url: url)
        guard precision != .float32 else { return Weights(arrays: raw) }
        // Cast only the floating-point tensors. An index or mask stored as an
        // integer stays what it is; casting one to bfloat16 would silently round
        // large indices.
        var cast = [String: MLXArray](minimumCapacity: raw.count)
        for (key, value) in raw {
            cast[key] = value.dtype == .float32 ? value.asType(precision.dtype) : value
        }
        return Weights(arrays: cast)
    }

    /// A view scoped to `path`, so a layer reads `"weight"` rather than its full path.
    public func scoped(_ path: String) -> Weights {
        Weights(arrays: arrays, prefix: prefix.isEmpty ? path : prefix + "." + path)
    }

    public func scoped(_ path: String, _ index: Int) -> Weights {
        scoped("\(path).\(index)")
    }

    public func has(_ key: String) -> Bool { arrays[fullKey(key)] != nil }

    public func callAsFunction(_ key: String) throws -> MLXArray {
        guard let a = arrays[fullKey(key)] else {
            throw WeightsError.missing(fullKey(key))
        }
        return a
    }

    /// `nil` rather than an error — for genuinely optional parameters such as the
    /// bias of a `Linear(bias: False)` or an `elementwise_affine: False` norm.
    public func optional(_ key: String) -> MLXArray? { arrays[fullKey(key)] }

    private func fullKey(_ key: String) -> String {
        prefix.isEmpty ? key : prefix + "." + key
    }
}

public enum WeightsError: Error, CustomStringConvertible {
    case missing(String)
    public var description: String {
        switch self {
        case .missing(let k): "checkpoint has no tensor '\(k)'"
        }
    }
}

// MARK: - Primitives

/// `torch.nn.Linear`: PyTorch stores `weight` as `[out, in]`, so the matmul
/// transposes it. Mirrors the reference exactly, bias included or absent.
@inlinable
public func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
    let y = MLX.matmul(x, w.transposed())
    return b.map { y + $0 } ?? y
}

/// `torch.nn.functional.rms_norm(x, (D,), weight, eps)`.
///
/// RMSNorm, not LayerNorm: the mean is never subtracted. Every norm in this trunk
/// is one of these with `eps = 1e-5`, and `AdaptiveLayerNorm.norm` is the only one
/// with no learned weight (`elementwise_affine: False`).
@inlinable
public func rmsNorm(_ x: MLXArray, _ weight: MLXArray?, eps: Float = 1e-5) -> MLXArray {
    let ms = MLX.mean(x * x, axis: -1, keepDims: true)
    let normed = x * MLX.rsqrt(ms + eps)
    return weight.map { normed * $0 } ?? normed
}

/// `torch.nn.LayerNorm`: subtracts the mean, unlike `rmsNorm`.
///
/// The two are NOT interchangeable and this network uses both. The trunk layers
/// normalise with RMSNorm; everything inside `PairReprUpdate` — including both
/// triangle-multiplication norms and the pair transition — uses LayerNorm. Several
/// of those carry a weight but no bias, hence the optional.
///
/// The variance is the biased (population) one, matching PyTorch.
@inlinable
public func layerNorm(
    _ x: MLXArray, _ weight: MLXArray?, _ bias: MLXArray?, eps: Float = 1e-5
) -> MLXArray {
    let mu = MLX.mean(x, axis: -1, keepDims: true)
    let centered = x - mu
    let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
    var y = centered * MLX.rsqrt(variance + eps)
    if let weight { y = y * weight }
    if let bias { y = y + bias }
    return y
}

/// Which normalisation a module was built with.
///
/// `make_norm` is a process-global factory read at construction time, so the same
/// module class normalises differently depending on when it was built. The released
/// bundle exercises both: the **flow trunk** uses RMSNorm, the **autoencoder** uses
/// LayerNorm. Porting either one hard-coded would silently corrupt the other.
public enum NormKind: Sendable { case rms, layer }

/// A normalisation layer loaded from the checkpoint, of either kind.
///
/// `elementwise_affine: False` appears as an absent weight rather than an error,
/// which is how `AdaptiveLayerNorm.norm` is stored.
public struct Norm {
    let kind: NormKind
    let weight: MLXArray?
    let bias: MLXArray?
    let eps: Float

    public init(_ w: Weights, kind: NormKind, eps: Float = 1e-5) {
        self.kind = kind
        self.weight = w.optional("weight")
        self.bias = w.optional("bias")
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch kind {
        case .rms: rmsNorm(x, weight, eps: eps)
        case .layer: layerNorm(x, weight, bias, eps: eps)
        }
    }
}

/// SwiGLU: split the last axis in half, gate one with SiLU.
/// `x, gates = x.chunk(2, -1); return silu(gates) * x`
@inlinable
public func swiGLU(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let a = x[.ellipsis, 0 ..< half]
    let gates = x[.ellipsis, half ..< (2 * half)]
    return (gates * MLX.sigmoid(gates)) * a
}
