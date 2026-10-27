import Foundation
import MLX
import RFD4Core

/// Flow-matching integration — `flow_matching/rdn_flow_matcher.py:simulation_step`
/// and `product_space_flow_matcher.py:full_simulation`.
///
/// Two modalities are integrated jointly, each on its own time grid and noise
/// schedule: `bb_ca` (the CA positions, 3-dim) and `local_latents` (the per-residue
/// 32-dim latent the decoder expands). The released configuration runs both in
/// `vf_tsr` mode — temporal score rescaling, made stochastic in the middle of the
/// trajectory and deterministic at both ends.
public enum Sampler {

    /// `vf_to_score(x_t, v, t) = (t * v - x_t) / (1 - t)`
    ///
    /// Asserts `t < 1` upstream, which the TSR-safe schedules honour by never
    /// reaching 1 until the final point — where the step is not evaluated.
    @inlinable
    public static func vfToScore(xT: MLXArray, v: MLXArray, t: Float) -> MLXArray {
        (t * v - xT) / (1 - t)
    }

    /// `score_to_vf(x_t, s, t) = (x_t + (1 - t) * s) / t`
    ///
    /// Divides by `t`, which is why the schedules start at `eps` rather than 0.
    @inlinable
    public static func scoreToVF(xT: MLXArray, score: MLXArray, t: Float) -> MLXArray {
        (xT + (1 - t) * score) / t
    }

    public struct StepParams: Sendable {
        public var samplingMode: SamplingMode
        public var scScaleNoise: Float
        public var tLimODE: Float
        public var tLimODEBelow: Float
        public var centerEveryStep: Bool
        public var tsrK: Float
        public var tsrSigma: Float

        public init(
            samplingMode: SamplingMode = .vfTSR, scScaleNoise: Float = 1.0,
            tLimODE: Float = 0.95, tLimODEBelow: Float = 0.02,
            centerEveryStep: Bool = false, tsrK: Float = 1.0, tsrSigma: Float = 1.0
        ) {
            self.samplingMode = samplingMode
            self.scScaleNoise = scScaleNoise
            self.tLimODE = tLimODE
            self.tLimODEBelow = tLimODEBelow
            self.centerEveryStep = centerEveryStep
            self.tsrK = tsrK
            self.tsrSigma = tsrSigma
        }

        /// `configs/generation/inference_base.yaml`.
        public static let releasedBackbone = StepParams(tsrK: 16.0, tsrSigma: 3.0)
        public static let releasedLocalLatents = StepParams(tsrK: 4.0, tsrSigma: 12.0)
        public static let released: [String: StepParams] = [
            "bb_ca": .releasedBackbone,
            "local_latents": .releasedLocalLatents,
        ]
    }

    /// Whether this step injects noise, i.e. takes the SDE branch.
    ///
    /// Upstream draws its noise INSIDE that branch, so a port that draws
    /// unconditionally advances its generator on steps where the reference does
    /// not. The distributions stay correct, but the two streams can never be
    /// aligned — which makes a matched-noise comparison impossible.
    @inlinable
    public static func isStochastic(t: Float, params: StepParams) -> Bool {
        switch params.samplingMode {
        case .vf: false
        case .sc: t <= params.tLimODE
        case .vfTSR: t != 0.0 && !(t > params.tLimODE || t < params.tLimODEBelow)
        }
    }

    /// One Euler step for one modality.
    ///
    /// - Parameters:
    ///   - v: the vector field the network predicted for this state
    ///   - t: the modality's time at this step (a scalar — upstream asserts every
    ///     sample shares it)
    ///   - dt: `ts[step + 1] - ts[step]`
    ///   - gt: the noise-injection schedule's value at this step
    ///   - noise: standard normal of `xT`'s shape, or `nil` for a deterministic step.
    ///     Passed in rather than drawn here so a caller can pin a seed — matched noise
    ///     is the only way to attribute a difference to precision rather than sampling.
    public static func step(
        xT: MLXArray, v: MLXArray, t: Float, dt: Float, gt: Float,
        params: StepParams, emitMask: MLXArray, noise: MLXArray? = nil
    ) -> MLXArray {
        var deltaX: MLXArray

        switch params.samplingMode {
        case .vf:
            deltaX = v * dt

        case .sc:
            // Close to t=1 the SDE stops and a low-temperature ODE finishes the
            // trajectory; 1.5 preserves that endpoint.
            let endpointScoreScale: Float = 1.5
            if t > params.tLimODE {
                let score = vfToScore(xT: xT, v: v, t: t) * endpointScoreScale
                deltaX = scoreToVF(xT: xT, score: score, t: t) * dt
            } else {
                let score = vfToScore(xT: xT, v: v, t: t)
                let stdEps = (2 * gt * params.scScaleNoise * dt).squareRoot()
                deltaX = (v + gt * score) * dt
                if let noise { deltaX = deltaX + stdEps * noise }
            }

        case .vfTSR:
            if t == 0.0 {
                // The schedules start at eps precisely so this branch is unreachable,
                // but it is upstream's guard against a division by zero and is kept.
                deltaX = MLX.zeros(like: v) * dt
            } else {
                let ratio = computeTSRRatio(t: [t], tsrK: params.tsrK,
                                            tsrSigma: params.tsrSigma)[0]
                let scaledScore = vfToScore(xT: xT, v: v, t: t) * ratio
                let vScaled = scoreToVF(xT: xT, score: scaledScore, t: t)
                if t > params.tLimODE || t < params.tLimODEBelow {
                    deltaX = vScaled * dt
                } else {
                    let stdEps = (2 * gt * params.scScaleNoise * dt).squareRoot()
                    deltaX = (vScaled + gt * scaledScore) * dt
                    if let noise { deltaX = deltaX + stdEps * noise }
                }
            }
        }

        var xNext = (xT + deltaX) * emitMask.expandedDimensions(axis: -1)
        if params.centerEveryStep { xNext = forceZeroCOM(xNext, emitMask: emitMask) }
        return xNext
    }

    /// Re-centre on the emitted tokens only, so padding cannot drag the centre.
    public static func forceZeroCOM(_ x: MLXArray, emitMask: MLXArray) -> MLXArray {
        let m = emitMask.expandedDimensions(axis: -1)
        let count = MLX.sum(m, axis: 1, keepDims: true)
        let centre = MLX.sum(x * m, axis: 1, keepDims: true) / MLX.maximum(count, 1)
        return (x - centre) * m
    }

    /// The integration loop.
    ///
    /// `predict` runs the network for one step and returns the vector field per
    /// modality. It is a closure rather than a stored `Trunk` because the features
    /// the trunk consumes have to be rebuilt from the current `x_t` at every step —
    /// which is the featurizer's job, not the sampler's.
    ///
    /// - Parameter noiseFor: supplies the per-step noise. Returning `nil` makes the
    ///   whole trajectory deterministic, which is what a matched-noise comparison or
    ///   a reproducible run wants.
    public static func simulate(
        x0: [String: MLXArray],
        schedule: FlowSchedule,
        nsteps: Int,
        recycleDepths: [Int],
        params: [String: StepParams],
        emitMasks: [String: MLXArray],
        predict: (_ x: [String: MLXArray], _ t: [String: Float], _ recycleDepth: Int)
            -> [String: MLXArray],
        noiseFor: ((_ dataMode: String, _ step: Int, _ shape: [Int]) -> MLXArray?)? = nil,
        onStep: ((_ step: Int, _ x: [String: MLXArray]) -> Void)? = nil
    ) -> [String: MLXArray] {
        var x = x0
        // Dictionary iteration order is NOT stable across processes in Swift, and
        // the modalities draw from a shared noise stream in this order. Iterating
        // the dictionary directly makes a seeded run unreproducible between runs,
        // which silently breaks any matched-noise comparison. Sort once, use
        // everywhere below.
        let modes = x.keys.sorted()

        for mode in modes where emitMasks[mode] != nil {
            x[mode] = x[mode]! * emitMasks[mode]!.expandedDimensions(axis: -1)
        }

        for stepIndex in 0 ..< nsteps {
            var t = [String: Float]()
            for mode in modes { t[mode] = schedule.ts[mode]![stepIndex] }

            let depth = stepIndex < recycleDepths.count ? recycleDepths[stepIndex] : 1
            let v = predict(x, t, depth)

            for mode in modes {
                guard let vm = v[mode], let ts = schedule.ts[mode],
                      let gts = schedule.gt[mode], let p = params[mode],
                      let emit = emitMasks[mode] else { continue }
                // Only draw where the step will actually use it, so this stream
                // has the same shape as the reference's.
                let noise = isStochastic(t: ts[stepIndex], params: p)
                    ? (noiseFor?(mode, stepIndex, x[mode]!.shape) ?? nil)
                    : nil
                x[mode] = step(
                    xT: x[mode]!, v: vm,
                    t: ts[stepIndex], dt: ts[stepIndex + 1] - ts[stepIndex],
                    gt: gts[stepIndex], params: p, emitMask: emit, noise: noise
                )
            }
            // Force the step before building the next one.
            //
            // MLX is lazy, so without this the whole trajectory accumulates as one
            // graph and nothing runs until a caller pulls a value out at the end.
            // Two things go wrong then, and the second is the dangerous one: a
            // progress callback reports steps that have not been computed, and
            // elapsed-time numbers measure graph construction rather than work —
            // this is why a 225-token binder appeared to generate in 0.4 s. The
            // graph also holds every intermediate of every step alive at once,
            // which is exactly the wrong memory profile for a phone.
            MLX.eval(modes.compactMap { x[$0] })
            onStep?(stepIndex, x)
        }
        return x
    }
}
