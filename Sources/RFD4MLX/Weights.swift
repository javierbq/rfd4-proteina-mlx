import Foundation
import MLX
import MLXFast

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
    /// Half precision halves the traffic over the pair track, which moves hundreds of
    /// megabytes per layer, and MLX accumulates matmuls in fp32 regardless of operand
    /// width. Measured on a HEAD-equivalent pair-update replica, uncontended:
    ///
    ///     n=320   fp32 213.5 ms   fp16 171.1 ms (1.25x)   bf16 171.0 ms (1.25x)
    ///     n=638   fp32 897.0 ms   fp16 727.3 ms (1.23x)   bf16 724.6 ms (1.24x)
    ///
    /// The speed is identical between the two half formats; the accuracy is not.
    /// After 7 stacked pair updates, against an fp32 reference:
    ///
    ///     fp16   max|d|/max|z| 2.2e-03    rms 9.6e-04
    ///     bf16   max|d|/max|z| 1.7e-02    rms 7.5e-03
    ///
    /// So **fp16 is the better half format here, by about 8x, for the same speed**.
    /// The usual reason to prefer bfloat16 — its wider exponent range — does not
    /// apply: the pair track's values sit around |z| < 3, nowhere near fp16's range
    /// limit, so fp16's extra mantissa bits are pure gain. Do not "upgrade" this to
    /// bfloat16 on general principle without re-measuring.
    ///
    /// None of this is exact, so the parity suite cannot clear it. Those numbers come
    /// from a 7-layer microbenchmark; a real run is 14 layers x 120 passes x 60 steps
    /// and the error compounds. The matched-noise harness is the instrument that
    /// settles whether a half format is good enough, by driving both precisions down
    /// upstream's own trajectory so what remains is precision rather than sampling.
    ///
    /// ## fp16 has now been validated end to end
    ///
    /// Driven down upstream's own noise, 60 steps, final state against upstream's:
    ///
    ///     de novo       fp32 rel 1.1e-06    fp16 rel 8.4e-04      (gate 5e-2)
    ///     binder        fp32 rel 1.4e-06    fp16 rel 6.4e-03      (gate 5e-2)
    ///
    /// And on the design campaign that matters — 24 seeds, one contig, the same
    /// binary, one precision each, paired on the seed:
    ///
    ///     per-design    fp32 104.3 s        fp16  80.1 s          1.30x
    ///     peak memory   fp32   2.31 GB      fp16   1.15 GB        2.01x
    ///     designable    fp32 21/24          fp16 21/24            differs at no seed
    ///
    /// 20 of those 24 seeds produce THE SAME DESIGN in both precisions — median
    /// 0.017 A CA RMSD, worst 0.101 A, 95-100% sequence identity. The other four
    /// land on a neighbouring sample (0.57-2.79 A), which is what a chaotic sampler
    /// does when a 1e-3 perturbation crosses a basin boundary; the oracle scores
    /// them as equivalent, and the one seed that fails designability fails in both.
    ///
    /// So fp16 is not bit-equivalent to fp32 and never will be. It is
    /// DISTRIBUTIONALLY equivalent, usually sample-identical, 1.3x faster, and half
    /// the peak memory — and on a machine where running out of GPU memory aborts
    /// the host process rather than throwing, halving the peak is a safety
    /// property, not an optimization.
    ///
    /// The one place the fixtures complain is the partial-diffusion trajectory,
    /// where fp16 exceeds the 5e-2 gate on local_latents (max rel 2.6e-01). That is
    /// 10 elements of 7200, with mean |d| 1.4e-03 and the backbone inside 0.13 A,
    /// and it does not survive the decoder: re-running partial diffusion in both
    /// precisions over fractions 0.3-0.9, three seeds each, agrees to 0.064 A CA
    /// RMSD at worst (median 0.029) and 98-100% sequence identity. The disagreement
    /// does not grow with the fraction — 0.9 re-runs three times as much of the
    /// trajectory as 0.3 and lands no further away — which is what rules out a
    /// compounding precision floor. The fixture tolerance is measuring the latent,
    /// not the structure.
    public enum Precision: String, Sendable {
        case float32, float16, bfloat16

        /// The default for a bare checkpoint, overridable with `RFD4_DTYPE` without
        /// a rebuild.
        ///
        /// fp16, on the evidence above: same designs, 1.3x faster, half the peak
        /// memory. `RFD4_DTYPE=fp32` opts back out, which is what to reach for when
        /// a result has to be reproducible against the fixtures rather than fast.
        ///
        /// This governs a bare `.safetensors` only. A pack declares its own
        /// precision and `RFD4Pack` passes that explicitly, so what is on disk is
        /// what gets loaded and this cannot reinterpret it.
        public static var fromEnvironment: Precision {
            switch ProcessInfo.processInfo.environment["RFD4_DTYPE"]?.lowercased() {
            case "fp32", "f32", "float32": return .float32
            case "bf16", "bfloat16": return .bfloat16
            case "fp16", "f16", "float16": return .float16
            default: return .float16
            }
        }

        /// The spelling a pack manifest uses, which is the short one the rest of
        /// this ecosystem writes — rfd3's manifest says `"fp32"`, not `"float32"`.
        /// Both spellings parse, so a manifest written either way loads.
        public init?(manifestName: String) {
            switch manifestName.lowercased() {
            case "fp32", "f32", "float32": self = .float32
            case "fp16", "f16", "float16": self = .float16
            case "bf16", "bfloat16": self = .bfloat16
            default: return nil
            }
        }

        /// What `init(manifestName:)` round-trips to, for an exporter writing one.
        public var manifestName: String {
            switch self {
            case .float32: "fp32"
            case .float16: "fp16"
            case .bfloat16: "bf16"
            }
        }

        public var dtype: DType {
            switch self {
            case .float32: .float32
            case .float16: .float16
            case .bfloat16: .bfloat16
            }
        }
    }

    public static func load(_ url: URL, precision: Precision = .fromEnvironment) throws
        -> Weights {
        Weights(arrays: narrowed(try MLX.loadArrays(url: url), to: precision))
    }

    /// Narrow a checkpoint's float tensors to `precision`, leaving everything else
    /// alone.
    ///
    /// Cast only the floating-point tensors. An index or mask stored as an integer
    /// stays what it is; casting one to bfloat16 would silently round large indices.
    ///
    /// This is a separate function so that the weight exporter can call it rather
    /// than reimplement it. A pack is supposed to be exactly what `load` would have
    /// produced in memory, written down — and two copies of this rule would be two
    /// chances for the file on disk to stop matching the thing it replaces.
    public static func narrowed(_ arrays: [String: MLXArray], to precision: Precision)
        -> [String: MLXArray] {
        guard precision != .float32 else { return arrays }
        var cast = [String: MLXArray](minimumCapacity: arrays.count)
        for (key, value) in arrays {
            cast[key] = value.dtype == .float32 ? value.asType(precision.dtype) : value
        }
        return cast
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

/// Variant of `linear` for weights already stored transposed — `[in, out]` — so the
/// matmul needs no further transpose. Use only with weights loaded via
/// `MLX.contiguous(rawW.transposed())` at construction time.
///
/// The reason to bother: `w.transposed()` returns a strided view, and MLX's Metal
/// matmul then selects the B-transposed GEMM path. Pre-materialising the transposed
/// layout lets the B-normal path run instead, which is measurably faster for the wide
/// pair-track matrices.
@inlinable
public func linearT(_ x: MLXArray, _ wT: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
    let y = MLX.matmul(x, wT)
    return b.map { y + $0 } ?? y
}

/// `torch.nn.functional.rms_norm(x, (D,), weight, eps)`.
///
/// RMSNorm, not LayerNorm: the mean is never subtracted. Every norm in this trunk
/// is one of these with `eps = 1e-5`, and `AdaptiveLayerNorm.norm` is the only one
/// with no learned weight (`elementwise_affine: False`).
@inlinable
public func rmsNorm(_ x: MLXArray, _ weight: MLXArray?, eps: Float = 1e-5) -> MLXArray {
    // MLXFast has a fused kernel; the arithmetic below is the same but reaches the
    // GPU as half a dozen separate elementwise passes over the whole tensor. On the
    // pair track that tensor is [b, n, n, 384] — 0.58 GB at n=638 — so the number of
    // passes over it is not a detail.
    if let weight { return MLXFast.rmsNorm(x, weight: weight, eps: eps) }
    let ms = MLX.mean(x * x, axis: -1, keepDims: true)
    return x * MLX.rsqrt(ms + eps)
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
    // Same reasoning as rmsNorm: the fused kernel replaces mean, subtract, square,
    // mean, rsqrt, scale and offset — seven passes — with one.
    MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
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
