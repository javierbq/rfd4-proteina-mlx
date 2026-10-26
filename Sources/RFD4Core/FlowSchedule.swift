import Foundation

// Flow-matching time and noise schedules.
//
// Ported from `flow_matching/product_space_flow_matcher.py` (`get_schedule`,
// `get_schedule_tsr_safe`, `_schedule_with_tail_multiplier`, `get_gt`,
// `ProductSpaceFlowMatcher.sample_schedule`) and `rdn_flow_matcher.py`
// (`compute_tsr_ratio`).
//
// All of this is pure numerics on 1-D arrays, so it lives in RFD4Core with no MLX
// dependency — the schedules are built once per run, on the host, before any tensor
// work starts.
//
// PRECISION: upstream computes these in float32 (torch's default), so this uses
// `Float`. That makes the arithmetic the same shape, but it does NOT prove bitwise
// agreement: `torch.linspace` and `torch.logspace` compute their step in double on
// CPU before casting, and this does not. Schedules are deterministic tensors, so the
// playbook wants them pinned bitwise -- that comparison needs dumped upstream
// fixtures and is Phase 3 work. Until then, treat these values as structurally
// correct and numerically unverified.

public enum ScheduleMode: String, Sendable, CaseIterable {
    case uniform, power, log
}

public enum GtMode: String, Sendable, CaseIterable {
    case ode
    case oneMinusTOverT = "1-t/t"
    case tan
    case oneOverT = "1/t"
}

public enum ScheduleError: Error, Equatable, CustomStringConvertible {
    case missingP1(ScheduleMode)
    case nonPositiveP1(ScheduleMode, Float)
    case tailStepsNotPositive
    case tailMultiplierNotInteger
    case nstepsTooSmallForTail
    case partialStepCountOutOfRange(nsteps: Int, fullNsteps: Int)
    case partialTimeMismatch(dataMode: String)

    public var description: String {
        switch self {
        case .missingP1(let m): "p1 cannot be none for the \(m.rawValue) schedule"
        case .nonPositiveP1(let m, let p): "p1 must be >0 for the \(m.rawValue) schedule, got \(p)"
        case .tailStepsNotPositive: "tail_steps must be positive"
        case .tailMultiplierNotInteger: "tail_step_multiplier must be an integer >= 2"
        case .nstepsTooSmallForTail: "nsteps must be at least tail_steps * tail_step_multiplier"
        case .partialStepCountOutOfRange(let n, let full):
            "Partial step count must be in [1, \(full)], got \(n)."
        case .partialTimeMismatch(let dm):
            "\(dm): partial corruption time must match the first time of the full schedule tail."
        }
    }
}

// MARK: - Grids

/// `torch.linspace(start, end, count)` in float32.
///
/// Reproduces PyTorch's `linspace_cpu` kernel, which accumulates forward from
/// `start` for the lower half of the grid and **backward from `end`** for the
/// upper half:
///
/// ```cpp
/// const int64_t halfway = steps / 2;
/// data[i] = i < halfway ? start + step * i
///                       : end - step * (steps - i - 1);
/// ```
///
/// The split is not incidental. It pins both endpoints exactly and keeps rounding
/// error symmetric instead of letting it accumulate across the whole grid — and
/// without it this port sits up to 40 ULP from upstream in the second half of every
/// schedule. With it, the residual is **at most 1 ULP** (~1.2e-7 relative), which is
/// torch's own rounding in that expression and is not recoverable without emulating
/// the kernel instruction-for-instruction. See `FlowScheduleParityTests`.
@inlinable
public func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
    guard count > 1 else { return count == 1 ? [start] : [] }
    let step = (end - start) / Float(count - 1)
    let halfway = count / 2
    return (0..<count).map { i in
        i < halfway ? start + step * Float(i)
                    : end - step * Float(count - i - 1)
    }
}

/// `torch.logspace(start, end, count)` — base 10 by default.
///
/// Shares `linspace`'s halfway split: torch's `logspace_cpu` is the same kernel with
/// a `pow` applied to each point.
@inlinable
public func logspace(_ start: Float, _ end: Float, _ count: Int, base: Float = 10) -> [Float] {
    linspace(start, end, count).map { pow(base, $0) }
}

// MARK: - Time schedules

/// Partition of the unit interval where the vector field / score is evaluated.
/// Returns `nsteps + 1` points.
public func getSchedule(
    mode: ScheduleMode,
    nsteps: Int,
    p1: Float? = nil,
    eps: Float = 1e-5,
    tailSteps: Int = 0,
    tailStepMultiplier: Int = 1
) throws -> [Float] {
    if tailSteps != 0 {
        return try scheduleWithTailMultiplier(nsteps: nsteps, tailSteps: tailSteps,
                                              multiplier: tailStepMultiplier) { steps in
            try getSchedule(mode: mode, nsteps: steps, p1: p1, eps: eps)
        }
    }
    switch mode {
    case .uniform:
        return linspace(0, 1, nsteps + 1)
    case .power:
        guard let p1 else { throw ScheduleError.missingP1(mode) }
        return linspace(0, 1, nsteps + 1).map { pow($0, p1) }
    case .log:
        guard let p1 else { throw ScheduleError.missingP1(mode) }
        guard p1 > 0 else { throw ScheduleError.nonPositiveP1(mode, p1) }
        var t = logspace(-p1, 0, nsteps + 1).reversed().map { 1.0 - $0 }
        let lo = t.min() ?? 0
        t = t.map { $0 - lo }
        let hi = t.max() ?? 1
        return t.map { $0 / hi }
    }
}

/// Schedule safe at `t = 0` when temporal score rescaling is used: starts at `eps`
/// rather than 0, because `score_to_vf` divides by `t`.
public func getScheduleTSRSafe(
    mode: ScheduleMode,
    nsteps: Int,
    p1: Float? = nil,
    eps: Float = 1e-3,
    tailSteps: Int = 0,
    tailStepMultiplier: Int = 1
) throws -> [Float] {
    if tailSteps != 0 {
        return try scheduleWithTailMultiplier(nsteps: nsteps, tailSteps: tailSteps,
                                              multiplier: tailStepMultiplier) { steps in
            try getScheduleTSRSafe(mode: mode, nsteps: steps, p1: p1, eps: eps)
        }
    }
    switch mode {
    case .uniform:
        return linspace(eps, 1, nsteps + 1)
    case .power:
        guard let p1 else { throw ScheduleError.missingP1(mode) }
        return linspace(pow(eps, 1 / p1), 1, nsteps + 1).map { pow($0, p1) }
    case .log:
        guard let p1 else { throw ScheduleError.missingP1(mode) }
        guard p1 > 0 else { throw ScheduleError.nonPositiveP1(mode, p1) }
        var t = logspace(-p1, eps, nsteps + 1).reversed().map { 1.0 - $0 }
        let lo = (t.min() ?? 0) - eps
        t = t.map { $0 - lo }
        let hi = t.max() ?? 1
        return t.map { $0 / hi }
    }
}

/// Sample the final `tailSteps` intervals at `multiplier` times the density.
///
/// Builds a schedule over `baseSteps * multiplier` fine steps, then keeps every
/// `multiplier`-th point up to the tail and every point within it. The result is
/// still `nsteps + 1` long.
public func scheduleWithTailMultiplier(
    nsteps: Int,
    tailSteps: Int,
    multiplier: Int,
    _ scheduleFn: (Int) throws -> [Float]
) throws -> [Float] {
    guard tailSteps >= 1 else { throw ScheduleError.tailStepsNotPositive }
    guard multiplier >= 2 else { throw ScheduleError.tailMultiplierNotInteger }

    let baseSteps = nsteps - (multiplier - 1) * tailSteps
    guard baseSteps >= tailSteps else { throw ScheduleError.nstepsTooSmallForTail }

    let fineSteps = baseSteps * multiplier
    let tailStart = (baseSteps - tailSteps) * multiplier
    let indices = Array(stride(from: 0, to: tailStart, by: multiplier)) + Array(tailStart...fineSteps)

    let fine = try scheduleFn(fineSteps)
    return indices.map { fine[$0] }
}

// MARK: - Noise injection

/// Noise injection schedule `g(t)` for the SDE
/// `dx_t = [v(x_t,t) + g(t) s(x_t,t)] dt + sqrt(2 g(t)) dw_t`.
public func getGt(
    t: [Float],
    mode: GtMode,
    param: Float,
    clampVal: Float? = nil,
    eps: Float = 1e-2
) -> [Float] {
    // Identically zero for all t, independent of param/clampVal: zeroes both the
    // Langevin drift and the noise term, recovering plain ODE integration.
    if mode == .ode { return [Float](repeating: 0, count: t.count) }

    let tc = t.map { min(max($0, 0), 1 - 1e-5) }  // for numerical reasons
    var gt: [Float]
    switch mode {
    case .ode:
        gt = []  // unreachable, handled above
    case .oneMinusTOverT:
        gt = tc.map { (1 - $0) / ($0 + eps) }
    case .tan:
        gt = tc.map {
            let a = (1 - $0) * Float.pi / 2
            return (Float.pi / 2) * sin(a) / (cos(a) + eps)
        }
    case .oneOverT:
        gt = tc.map { 1 / ($0 + eps) }
    }

    gt = transformGt(gt, fPow: param)
    if let clampVal { gt = gt.map { min(max($0, 0), clampVal) } }
    else { gt = gt.map { max($0, 0) } }
    return gt
}

/// Reshape `gt` by `fPow`, log-normalising around its own mean first.
/// `fPow == 1` is the identity.
private func transformGt(_ gt: [Float], fPow: Float) -> [Float] {
    guard fPow != 1.0, !gt.isEmpty else { return gt }
    let logGt = gt.map { log($0) }
    let meanLogGt = logGt.reduce(0, +) / Float(logGt.count)
    return logGt.map { lg in
        let normalized = pow(1 / (1 + exp(-(lg - meanLogGt))), fPow)
        // torch.logit(x, eps=1e-6) clamps to [eps, 1-eps] before log(x/(1-x)).
        let c = min(max(normalized, 1e-6), 1 - 1e-6)
        return exp(log(c / (1 - c)) + meanLogGt)
    }
}

/// Temporal-score-rescaling low-temperature factor (`rdn_flow_matcher.py:553`):
///
/// ```python
/// snr = t**2 / (1 - t) ** 2
/// return (snr * tsr_sigma**2 + 1) / (snr * tsr_sigma**2 / tsr_k + 1)
/// ```
///
/// `tsr_sigma**2` is a scalar Python expression evaluated **once** before it ever
/// meets the tensor, so it must be pre-squared here too rather than multiplied in
/// elementwise twice. Writing it as `snr * sigma * sigma` costs 1 ULP per element,
/// which the squared division in `sampleSchedule` then amplifies — it was worth
/// 125 ULP on `local_latents.gt`. Pre-squaring makes the ratio bit-exact.
public func computeTSRRatio(t: [Float], tsrK: Float, tsrSigma: Float) -> [Float] {
    let sigmaSquared = tsrSigma * tsrSigma
    return t.map { ti in
        let snr = (ti * ti) / ((1 - ti) * (1 - ti))
        return (snr * sigmaSquared + 1) / (snr * sigmaSquared / tsrK + 1)
    }
}

// MARK: - Per-modality configuration

public struct ModalitySchedule: Sendable {
    public var mode: ScheduleMode
    public var p: Float
    public var tailSteps: Int
    public var tailStepMultiplier: Int

    public init(mode: ScheduleMode, p: Float, tailSteps: Int = 0, tailStepMultiplier: Int = 1) {
        self.mode = mode
        self.p = p
        self.tailSteps = tailSteps
        self.tailStepMultiplier = tailStepMultiplier
    }
}

public struct ModalityGt: Sendable {
    public var mode: GtMode
    public var p: Float
    public var clampVal: Float?

    public init(mode: GtMode, p: Float, clampVal: Float? = nil) {
        self.mode = mode
        self.p = p
        self.clampVal = clampVal
    }
}

public enum SamplingMode: String, Sendable {
    /// plain flow matching
    case vf
    /// score-based
    case sc
    /// temporal score rescaling — the release default for both modalities
    case vfTSR = "vf_tsr"
}

public struct ModalityConfig: Sendable {
    public var schedule: ModalitySchedule
    public var gt: ModalityGt
    public var samplingMode: SamplingMode
    public var tsrK: Float
    public var tsrSigma: Float

    public init(schedule: ModalitySchedule, gt: ModalityGt,
                samplingMode: SamplingMode = .vfTSR, tsrK: Float = 1, tsrSigma: Float = 1) {
        self.schedule = schedule
        self.gt = gt
        self.samplingMode = samplingMode
        self.tsrK = tsrK
        self.tsrSigma = tsrSigma
    }

    /// `configs/generation/inference_base.yaml`, both modalities.
    public static let releasedBackbone = ModalityConfig(
        schedule: ModalitySchedule(mode: .log, p: 2.0, tailSteps: 10, tailStepMultiplier: 2),
        gt: ModalityGt(mode: .oneOverT, p: 1.0, clampVal: nil),
        samplingMode: .vfTSR, tsrK: 16.0, tsrSigma: 3.0
    )
    public static let releasedLocalLatents = ModalityConfig(
        schedule: ModalitySchedule(mode: .power, p: 2.0, tailSteps: 10, tailStepMultiplier: 2),
        gt: ModalityGt(mode: .tan, p: 1.0, clampVal: nil),
        samplingMode: .vfTSR, tsrK: 4.0, tsrSigma: 12.0
    )

    public static let releasedDataModes = ["bb_ca", "local_latents"]
    public static let released: [String: ModalityConfig] = [
        "bb_ca": .releasedBackbone,
        "local_latents": .releasedLocalLatents,
    ]
}

/// A built schedule: `ts` has `nsteps + 1` points per modality, `gt` has `nsteps`.
public struct FlowSchedule: Sendable {
    public var ts: [String: [Float]]
    public var gt: [String: [Float]]
}

/// Per-modality time and noise schedules (`ProductSpaceFlowMatcher.sample_schedule`).
public func sampleSchedule(
    nsteps: Int,
    modalities: [String: ModalityConfig]
) throws -> FlowSchedule {
    var ts = [String: [Float]](), gt = [String: [Float]]()

    for (name, cfg) in modalities {
        let builder = cfg.samplingMode == .vfTSR ? getScheduleTSRSafe : getSchedule
        ts[name] = try builder(cfg.schedule.mode, nsteps, cfg.schedule.p,
                               cfg.samplingMode == .vfTSR ? 1e-3 : 1e-5,
                               cfg.schedule.tailSteps, cfg.schedule.tailStepMultiplier)
    }
    for (name, cfg) in modalities {
        // ts[-1] == 1 is never evaluated, so gt is one shorter.
        let head = Array(ts[name]!.dropLast())
        var g = getGt(t: head, mode: cfg.gt.mode, param: cfg.gt.p, clampVal: cfg.gt.clampVal)
        if cfg.samplingMode == .vfTSR {
            // Matched Langevin drift and noise against the TSR-rescaled score.
            let ratio = computeTSRRatio(t: head, tsrK: cfg.tsrK, tsrSigma: cfg.tsrSigma)
            g = zip(g, ratio).map { $0 / ($1 * $1) }
        }
        gt[name] = g
    }
    return FlowSchedule(ts: ts, gt: gt)
}

/// Per-modality schedules covering `[partialT, 1.0]` in exactly `nsteps` steps,
/// by slicing the tail of the configured full schedule
/// (`search/vanilla.py:build_partial_schedule`, `full_nsteps` branch).
///
/// Partial diffusion resumes a trajectory that was noised to `partialT`, so the
/// integration must start exactly where the state was noised — hence the equality
/// check rather than a tolerance.
public func buildPartialSchedule(
    partialT: [String: Float],
    nsteps: Int,
    fullNsteps: Int,
    modalities: [String: ModalityConfig]
) throws -> FlowSchedule {
    guard (1...fullNsteps).contains(nsteps) else {
        throw ScheduleError.partialStepCountOutOfRange(nsteps: nsteps, fullNsteps: fullNsteps)
    }
    let full = try sampleSchedule(nsteps: fullNsteps, modalities: modalities)

    var ts = [String: [Float]](), gt = [String: [Float]]()
    for name in modalities.keys {
        ts[name] = Array(full.ts[name]!.suffix(nsteps + 1))
        gt[name] = Array(full.gt[name]!.suffix(nsteps))
    }
    for name in modalities.keys {
        guard let start = partialT[name], ts[name]![0] == start else {
            throw ScheduleError.partialTimeMismatch(dataMode: name)
        }
    }
    return FlowSchedule(ts: ts, gt: gt)
}
