import Foundation

/// One `[width, depth]` band of a recycle schedule: `width` consecutive sampling
/// steps, each run at `depth` recycles.
public struct RecycleBand: Equatable, Sendable {
    public let width: Int
    public let depth: Int

    public init(width: Int, depth: Int) {
        self.width = width
        self.depth = depth
    }
}

/// Inference-time recycling configuration — the subset of upstream's `inf_cfg`
/// that `Proteina.recycle_depths` reads.
///
/// Upstream carries this as a loose dict; here the two mutually exclusive forms are
/// a sum type, so "schedule present" and "constant depth" cannot both be set.
public enum RecycleConfig: Sendable {
    /// A schedule defined over the full trajectory.
    case schedule([RecycleBand])
    /// One depth for every step actually run. Upstream marks this
    /// `TODO: Deprecate this option`.
    case constant(Int)
}

public enum RecycleScheduleError: Error, Equatable, CustomStringConvertible {
    case requiresRecyclingCheckpoint
    case lengthMismatch(expands: Int, expected: Int)
    case negativeDepth

    public var description: String {
        switch self {
        case .requiresRecyclingCheckpoint:
            "n_recycle_schedule requires a recycling checkpoint"
        case .lengthMismatch(let expands, let expected):
            "recycle schedule expands to \(expands) steps; expected \(expected)"
        case .negativeDepth:
            "recycle depths must be non-negative integers"
        }
    }
}

extension RecycleConfig {
    /// Recycle depth for each of the `nsteps` sampling steps actually run.
    ///
    /// Ported from `proteina.py:Proteina.recycle_depths`.
    ///
    /// A schedule is defined over the **full** trajectory. Partial diffusion runs
    /// only its tail, so when `nsteps < fullNsteps` the schedule is expanded over
    /// `fullNsteps` and its final `nsteps` depths are returned — it must not raise on
    /// the length difference. `fullNsteps` defaults to `nsteps`, i.e. a full
    /// trajectory.
    ///
    /// The compute unit this produces is the real one: the released 60-step schedules
    /// expand to 120–300 trunk passes, not 60.
    public func depths(
        nsteps: Int,
        fullNsteps: Int? = nil,
        recycleEnabled: Bool = true
    ) throws -> [Int] {
        let fullNsteps = fullNsteps ?? nsteps
        let depths: [Int]

        switch self {
        case .constant(let depth):
            depths = [Int](repeating: depth, count: max(0, nsteps))

        case .schedule(let bands):
            guard recycleEnabled else { throw RecycleScheduleError.requiresRecyclingCheckpoint }
            // A non-positive width contributes nothing, matching `range(width)`.
            let expanded = bands.flatMap { [Int](repeating: $0.depth, count: max(0, $0.width)) }
            guard expanded.count == fullNsteps else {
                throw RecycleScheduleError.lengthMismatch(expands: expanded.count,
                                                          expected: fullNsteps)
            }
            depths = Array(expanded.suffix(nsteps))
        }

        guard depths.allSatisfy({ $0 >= 0 }) else { throw RecycleScheduleError.negativeDepth }
        return depths
    }

    /// Total trunk passes a run costs — the number the time and memory budget should
    /// actually be built from.
    public func trunkPasses(nsteps: Int, fullNsteps: Int? = nil) throws -> Int {
        // A depth of 0 still runs the trunk once; depth counts *re*cycles.
        try depths(nsteps: nsteps, fullNsteps: fullNsteps).reduce(0) { $0 + max(1, $1) }
    }
}

/// The recycle schedules shipped with the release, transcribed from
/// `configs/recycle_strategy/*/front_loaded.yaml`.
public enum ReleasedRecycleStrategy: String, CaseIterable, Sendable {
    case thinkingLow = "thinking_low"
    case thinkingMedium = "thinking_medium"
    case thinkingHigh = "thinking_high"

    public var bands: [RecycleBand] {
        switch self {
        case .thinkingLow:
            [(10, 4), (10, 3), (10, 2), (30, 1)].map(RecycleBand.init)
        case .thinkingMedium:
            [(5, 6), (10, 5), (5, 4), (10, 3), (20, 2), (10, 1)].map(RecycleBand.init)
        case .thinkingHigh:
            [(10, 9), (10, 8), (10, 5), (20, 3), (10, 2)].map(RecycleBand.init)
        }
    }

    public var config: RecycleConfig { .schedule(bands) }

    /// Every released strategy is defined over a 60-step trajectory.
    public static let nsteps = 60
}
