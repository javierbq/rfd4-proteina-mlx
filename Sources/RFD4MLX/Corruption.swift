import Foundation
import MLX
import MLXRandom
import RFD4Core

/// The forward process: take a clean state to a chosen time on the trajectory.
///
/// `x_t = (1 − t) · x_0 + t · x_1`, with both endpoints emit-masked. That part is a
/// plain interpolation. What is not plain is `x_0`, and getting it wrong would be
/// invisible — the result is still Gaussian-looking noise of about the right size,
/// and only the *distribution* is off, which no single sample reveals.
///
/// Measured off the loaded checkpoint rather than read from a training config the
/// released weights may not have used:
///
///   `bb_ca`           `zero_com_noise` true, then stochastic centering in mode
///                     `x_0`, scope `per_group` over `molecule_iid`, scale 0.2
///   `local_latents`   neither; a plain masked Gaussian
///
/// So a two-chain complex gets **two independent random translations**, one per
/// molecule, on top of zero-centred noise. Omitting that would corrupt the two
/// chains as a rigid pair and quietly narrow the distribution the model was
/// calibrated against.
public enum Corruption {

    /// Per-modality forward-process settings.
    public struct Settings: Sendable {
        public var zeroCOMNoise: Bool
        /// `nil` when the mode is `none`; otherwise the translation scale.
        public var stochasticCenteringScale: Float?

        public init(zeroCOMNoise: Bool, stochasticCenteringScale: Float?) {
            self.zeroCOMNoise = zeroCOMNoise
            self.stochasticCenteringScale = stochasticCenteringScale
        }

        /// What the released checkpoint carries.
        public static let released: [String: Settings] = [
            "bb_ca": Settings(zeroCOMNoise: true, stochasticCenteringScale: 0.2),
            "local_latents": Settings(zeroCOMNoise: false, stochasticCenteringScale: nil),
        ]
    }

    /// An independent random translation per group.
    ///
    /// One `[d]` Gaussian per distinct group id among the eligible tokens, scaled and
    /// added to every token of that group. The groups are `molecule_iid`, so for a
    /// binder this is one translation for the design and one for the target.
    ///
    /// - Parameters:
    ///   - groupIDs: `[b, n]` group label per token.
    ///   - offsetMask: `[b, n]` which tokens may be moved.
    ///   - draw: supplies each group's `[d]` Gaussian, in ascending group order —
    ///     a parameter rather than an internal draw so a matched-noise comparison
    ///     can replay upstream's own numbers.
    public static func stochasticOffset(
        _ x: MLXArray,
        groupIDs: MLXArray,
        offsetMask: MLXArray,
        scale: Float,
        draw: (_ group: Int, _ dim: Int) -> MLXArray
    ) -> MLXArray {
        let eligible = offsetMask .> 0
        // Ascending order, and only groups that actually have an eligible token:
        // upstream iterates `group_ids[offset_mask].unique()`, which is sorted and
        // excludes groups wholly outside the mask. Drawing for a group that is not
        // there would consume a Gaussian upstream never drew and shift every
        // subsequent draw in a matched-noise comparison.
        let ids = groupIDs.asArray(Float.self)
        let mask = eligible.asArray(Bool.self)
        var present = Set<Int>()
        for (i, m) in mask.enumerated() where m { present.insert(Int(ids[i])) }

        var out = x
        for group in present.sorted() {
            let inGroup = MLX.logicalAnd(eligible, groupIDs .== Float(group))
            let gate = inGroup.asType(.float32).expandedDimensions(axis: -1)
            let offset = draw(group, x.dim(-1)).reshaped(1, 1, x.dim(-1))
            out = out + scale * offset * gate
        }
        return out
    }

    /// `x_t = (1 − t) · x_0 + t · x_1`, both endpoints emit-masked.
    public static func interpolate(
        x0: MLXArray, x1: MLXArray, t: Float, emitMask: MLXArray
    ) -> MLXArray {
        let m = emitMask.expandedDimensions(axis: -1)
        return (1 - t) * (x0 * m) + t * (x1 * m)
    }

    public struct Corrupted {
        /// The reference sample as `corrupt_batch` records it — **before** the
        /// per-group translations.
        ///
        /// Upstream applies those inside `interpolate`, to a local variable, and
        /// stores this one in the batch. So `x_t` is not `lerp(x0, x1, t)`: measured
        /// on a two-chain complex at t = 0.97, the two differ by 9.3e-03, which is
        /// `(1 - t)` times the offsets. Treating the stored `x_0` as the thing that
        /// was interpolated is a plausible reading that quietly does not reconstruct
        /// `x_t`, so both are returned rather than one.
        public let x0: [String: MLXArray]
        /// What the interpolation actually used — `x0` plus the translations.
        public let x0Interpolated: [String: MLXArray]
        /// The corrupted state the partial trajectory restarts at.
        public let xt: [String: MLXArray]
    }

    /// Take a clean state to `t`, drawing exactly what upstream draws, in order.
    ///
    /// The order is not incidental. `corrupt_batch` calls `sample_noise` for **every**
    /// modality and only then `interpolate`, and the per-group translations happen
    /// inside `interpolate` — so the base noises come first, both of them, and the
    /// offsets after. Measured: four draws for a two-chain complex, shaped
    /// `[1,n,3]`, `[1,n,32]`, `[1,3]`, `[1,3]`. Applying each modality's offsets
    /// immediately after its own base noise would consume the same count in a
    /// different order and silently decorrelate a matched-noise replay.
    ///
    /// - Parameters:
    ///   - baseNoise: `[b, n, d]` per modality, in ascending modality order.
    ///   - offsetDraw: `[b, d]` per (modality, group), after all the base noise.
    public static func corrupt(
        x1: [String: MLXArray],
        t: [String: Float],
        emitMasks: [String: MLXArray],
        settings: [String: Settings] = Settings.released,
        groupIDs: MLXArray?,
        baseNoise: (_ mode: String, _ shape: [Int]) -> MLXArray,
        offsetDraw: (_ mode: String, _ group: Int, _ dim: Int) -> MLXArray
    ) -> Corrupted {
        // Sorted, because Swift `Dictionary` order is not stable across processes
        // and these draws come off one stream. The de-novo sampler learned that.
        let modes = x1.keys.sorted()

        var x0 = [String: MLXArray]()
        for mode in modes {
            let emit = emitMasks[mode]!
            let shape = x1[mode]!.shape
            var v = baseNoise(mode, shape) * emit.expandedDimensions(axis: -1)
            if settings[mode]?.zeroCOMNoise == true {
                v = Sampler.forceZeroCOM(v, emitMask: emit)
            }
            x0[mode] = v
        }

        var shifted = x0
        for mode in modes {
            guard let scale = settings[mode]?.stochasticCenteringScale,
                  let groupIDs else { continue }
            shifted[mode] = stochasticOffset(
                x0[mode]!, groupIDs: groupIDs, offsetMask: emitMasks[mode]!,
                scale: scale, draw: { offsetDraw(mode, $0, $1) })
        }

        var xt = [String: MLXArray]()
        for mode in modes {
            xt[mode] = interpolate(x0: shifted[mode]!, x1: x1[mode]!,
                                   t: t[mode]!, emitMask: emitMasks[mode]!)
        }
        return Corrupted(x0: x0, x0Interpolated: shifted, xt: xt)
    }
}
