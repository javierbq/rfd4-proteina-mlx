import Testing
@testable import RFD4Core

/// Ported from upstream `tests/test_recycle_depths.py`.
///
/// Upstream builds a bare `Proteina` with `__new__` and stubs `inf_cfg` / `nn`.
/// Here the configuration is its own value type, so there is nothing to stub —
/// `RecycleConfig.depths(nsteps:fullNsteps:recycleEnabled:)` is the whole unit.
@Suite("recycle depths")
struct RecycleDepthsTests {
    /// Released 1x schedule: `[width, depth]` bands summing to 60 evaluations.
    static let schedule = RecycleConfig.schedule(
        [(5, 1), (5, 2), (5, 3), (5, 5), (10, 6), (5, 5), (5, 3), (10, 2), (10, 1)]
            .map(RecycleBand.init)
    )

    // MARK: test_full_trajectory_expands_whole_schedule

    @Test("a full trajectory expands the whole schedule")
    func fullTrajectoryExpandsWholeSchedule() throws {
        let depths = try Self.schedule.depths(nsteps: 60, fullNsteps: 60)
        #expect(depths.count == 60)
        // first band, peak band, last band
        #expect(depths[0] == 1)
        #expect(depths[20] == 6)
        #expect(depths[depths.count - 1] == 1)
    }

    // MARK: test_partial_diffusion_takes_schedule_tail

    @Test("partial diffusion takes the schedule tail")
    func partialDiffusionTakesScheduleTail() throws {
        // Partial diffusion runs the final `nsteps` of the trajectory, so this returns
        // the tail of the full schedule. It must NOT raise on the length difference.
        let full = try Self.schedule.depths(nsteps: 60, fullNsteps: 60)
        let partial = try Self.schedule.depths(nsteps: 15, fullNsteps: 60)
        #expect(partial.count == 15)
        #expect(partial == Array(full.suffix(15)))
    }

    // MARK: test_full_nsteps_defaults_to_nsteps

    @Test("fullNsteps defaults to nsteps")
    func fullNstepsDefaultsToNsteps() throws {
        #expect(try Self.schedule.depths(nsteps: 60)
                == Self.schedule.depths(nsteps: 60, fullNsteps: 60))
    }

    // MARK: test_schedule_length_must_match_full_nsteps

    @Test("a schedule that does not expand to fullNsteps is rejected")
    func scheduleLengthMustMatchFullNsteps() {
        #expect(throws: RecycleScheduleError.lengthMismatch(expands: 60, expected: 50)) {
            try Self.schedule.depths(nsteps: 40, fullNsteps: 50)
        }
        // Upstream matches on the message text, so pin it too.
        #expect(RecycleScheduleError.lengthMismatch(expands: 60, expected: 50).description
                == "recycle schedule expands to 60 steps; expected 50")
    }

    // MARK: test_no_schedule_uses_constant_depth_per_step_run

    @Test("no schedule means a constant depth per step run")
    func noScheduleUsesConstantDepth() throws {
        #expect(try RecycleConfig.constant(2).depths(nsteps: 15)
                == [Int](repeating: 2, count: 15))
    }

    // MARK: test_schedule_requires_recycling_checkpoint

    @Test("a schedule requires a recycling checkpoint")
    func scheduleRequiresRecyclingCheckpoint() {
        #expect(throws: RecycleScheduleError.requiresRecyclingCheckpoint) {
            try Self.schedule.depths(nsteps: 60, fullNsteps: 60, recycleEnabled: false)
        }
    }

    // MARK: released strategies (not upstream; pins the shipped configs)

    @Test("released strategies expand to 60 steps and their documented trunk-pass cost",
          arguments: [
            (ReleasedRecycleStrategy.thinkingLow, 120),
            (ReleasedRecycleStrategy.thinkingMedium, 180),
            (ReleasedRecycleStrategy.thinkingHigh, 300),
          ])
    func releasedStrategies(strategy: ReleasedRecycleStrategy, expectedPasses: Int) throws {
        let n = ReleasedRecycleStrategy.nsteps
        let depths = try strategy.config.depths(nsteps: n, fullNsteps: n)
        #expect(depths.count == n, "\(strategy.rawValue) must cover all 60 steps")
        // The compute unit is trunk passes, not sampling steps.
        #expect(try strategy.config.trunkPasses(nsteps: n, fullNsteps: n) == expectedPasses)
    }

    @Test("front-loaded strategies never increase in depth")
    func strategiesAreFrontLoaded() throws {
        for strategy in ReleasedRecycleStrategy.allCases {
            let depths = try strategy.config.depths(nsteps: 60, fullNsteps: 60)
            #expect(zip(depths, depths.dropFirst()).allSatisfy { $0 >= $1 },
                    "\(strategy.rawValue) is not monotonically non-increasing")
        }
    }
}
