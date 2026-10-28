import Foundation

/// Where on the trajectory a partial-diffusion run restarts, and the tail it runs.
///
/// Starting from a structure instead of from noise means picking a corruption level
/// and then denoising only what remains. The arithmetic is small but every part of
/// it is load-bearing, and the fractional form in particular is easy to read as
/// something it is not.
///
/// **A fraction is a step count, not a time.** `partial_trajectory_fraction_steps`
/// names how many of the configured schedule's intervals to run; the corruption time
/// is then read off that schedule at the restart index. Upstream says it plainly —
/// "a restart index is not a diffusion time: changing the schedule or its step count
/// can change the corruption level at a given fraction" — and the consequence is
/// that this cannot be computed from the fraction alone. It needs the schedule.
public enum PartialDiffusion {

    /// How many of the full trajectory's steps a fraction asks for.
    ///
    /// Rounded to nearest, and at least one: a fraction small enough to round to
    /// zero is an error rather than a no-op run, because a run that denoises nothing
    /// returns the corrupted input and looks like a very conservative design.
    public static func remainingSteps(fraction: Double, nsteps: Int) throws -> Int {
        guard fraction > 0, fraction <= 1 else {
            throw PartialDiffusionError.fractionOutOfRange(fraction)
        }
        let remaining = Int((fraction * Double(nsteps)).rounded())
        guard remaining >= 1 else {
            throw PartialDiffusionError.fractionRoundsToZero(fraction: fraction,
                                                             nsteps: nsteps)
        }
        return remaining
    }

    /// The index into the full schedule that a fractional restart begins at.
    public static func restartIndex(fraction: Double, nsteps: Int) throws -> Int {
        nsteps - (try remainingSteps(fraction: fraction, nsteps: nsteps))
    }

    /// The last `nsteps` intervals of a full schedule.
    ///
    /// `ts` keeps `nsteps + 1` points and `gt` keeps `nsteps`, so the two slices are
    /// of different lengths — `gt` is indexed by interval, `ts` by endpoint.
    public static func tail(of schedule: FlowSchedule, nsteps: Int) throws -> FlowSchedule {
        var ts = [String: [Float]](), gt = [String: [Float]]()
        for (mode, full) in schedule.ts {
            let fullSteps = full.count - 1
            guard nsteps >= 1, nsteps <= fullSteps else {
                throw PartialDiffusionError.stepCountOutOfRange(nsteps: nsteps,
                                                                full: fullSteps)
            }
            ts[mode] = Array(full.suffix(nsteps + 1))
        }
        for (mode, full) in schedule.gt {
            gt[mode] = Array(full.suffix(nsteps))
        }
        return FlowSchedule(ts: ts, gt: gt)
    }

    /// The corruption time per modality for a fractional restart.
    ///
    /// Read off the **full** schedule at the restart index, which is also the first
    /// point of the tail — upstream asserts the two agree, and so does this.
    public static func partialT(
        fraction: Double, nsteps: Int, schedule: FlowSchedule
    ) throws -> [String: Float] {
        let k = try restartIndex(fraction: fraction, nsteps: nsteps)
        var out = [String: Float]()
        for (mode, full) in schedule.ts {
            guard k < full.count else {
                throw PartialDiffusionError.stepCountOutOfRange(nsteps: nsteps,
                                                                full: full.count - 1)
            }
            let t = full[k]
            // t = 1 is the clean state and the score conversion divides by 1 - t.
            guard t < 1 else { throw PartialDiffusionError.startTimeAtOrAboveOne(mode, t) }
            out[mode] = t
        }
        return out
    }
}

public enum PartialDiffusionError: Error, Equatable, CustomStringConvertible {
    case fractionOutOfRange(Double)
    case fractionRoundsToZero(fraction: Double, nsteps: Int)
    case stepCountOutOfRange(nsteps: Int, full: Int)
    case startTimeAtOrAboveOne(String, Float)

    public var description: String {
        switch self {
        case .fractionOutOfRange(let f):
            return "partial trajectory fraction must be in (0, 1], got \(f)."
        case .fractionRoundsToZero(let f, let n):
            return """
                a fraction of \(f) over \(n) steps rounds to zero denoising steps, \
                which would return the corrupted input unchanged. Raise the fraction \
                or the step count.
                """
        case .stepCountOutOfRange(let n, let full):
            return "partial step count must be in [1, \(full)], got \(n)."
        case .startTimeAtOrAboveOne(let mode, let t):
            return """
                \(mode) would restart at t = \(t). t = 1 is the clean state and the \
                score conversion divides by 1 − t, so the start time must be below it.
                """
        }
    }
}
