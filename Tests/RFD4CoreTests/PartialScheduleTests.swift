import Testing
@testable import RFD4Core

/// Ported from upstream `tests/test_partial_schedule.py`.
///
/// Upstream docstring: "The partial-diffusion schedule must start where the state was
/// actually noised."
///
/// SCOPE. This covers the schedule math and the `full_nsteps` branch of
/// `build_partial_schedule`. Three upstream tests are NOT yet ported and are tracked
/// in `docs/test-port-manifest.md`:
///   - `test_schedule_starts_at_the_noised_time` and
///     `test_fraction_corruption_uses_configured_grid` need `resolve_partial_t`,
///     which reads the inference config and the inverse-folding switches.
///   - `test_fractional_generation_corrupts_at_first_integration_time` needs the
///     Vanilla/Beam search strategies (Phase 2c).
///
/// NOTE ON EXACTNESS. Upstream compares full-vs-partial slices with
/// `rtol=0, atol=0`. That is a self-consistency check — both sides come from the same
/// builder — so it ports exactly and is kept exact here. It does NOT establish
/// agreement with upstream's own float32 values; see the precision note in
/// `FlowSchedule.swift`.
@Suite("partial schedule")
struct PartialScheduleTests {
    static let modalities = ModalityConfig.released
    static let dataModes = ModalityConfig.releasedDataModes

    // MARK: test_fraction_preserves_full_schedule_tail

    /// Cartesian product of upstream's three parameters. `noisePower` is the only
    /// thing that exercises `transformGt`'s non-identity branch, so it earns its place.
    static let tailCases: [(tailSteps: Int, noisePower: Float, remainingSteps: Int)] = {
        var cases: [(Int, Float, Int)] = []
        for tail in [0, 10] {
            for power in [Float(1.0), Float(1.5)] {
                for remaining in [1, 2, 20, 21, 28, 60] {
                    cases.append((tail, power, remaining))
                }
            }
        }
        return cases
    }()

    @Test("a fractional restart preserves the full schedule tail", arguments: tailCases)
    func fractionPreservesFullScheduleTail(
        tailSteps: Int, noisePower: Float, remainingSteps: Int
    ) throws {
        var modalities = Self.modalities
        for (name, var cfg) in modalities {
            cfg.schedule.tailSteps = tailSteps
            cfg.gt.p = noisePower
            modalities[name] = cfg
        }

        let full = try sampleSchedule(nsteps: 60, modalities: modalities)
        let partialT = Dictionary(uniqueKeysWithValues: Self.dataModes.map {
            ($0, full.ts[$0]![full.ts[$0]!.count - remainingSteps - 1])
        })
        let partial = try buildPartialSchedule(partialT: partialT, nsteps: remainingSteps,
                                               fullNsteps: 60, modalities: modalities)

        for dm in Self.dataModes {
            let ts = try #require(partial.ts[dm]), gt = try #require(partial.gt[dm])
            // Exact: both sides come from the same builder.
            #expect(ts == Array(full.ts[dm]!.suffix(remainingSteps + 1)))
            #expect(gt == Array(full.gt[dm]!.suffix(remainingSteps)))
            #expect(ts[0] == partialT[dm])
            #expect(ts[ts.count - 1] == 1.0, "a partial trajectory must end at t=1")
            #expect(zip(ts, ts.dropFirst()).allSatisfy { $0 < $1 },
                    "\(dm): times must be strictly increasing")
            #expect(gt.allSatisfy { $0.isFinite }, "\(dm): gt must be finite")
            #expect(gt.count == ts.count - 1)
        }
    }

    // MARK: test_fraction_rejects_mismatched_corruption_time

    @Test("a corruption time that is not on the schedule is rejected")
    func rejectsMismatchedCorruptionTime() {
        #expect(throws: ScheduleError.self) {
            try buildPartialSchedule(partialT: ["bb_ca": 0.5, "local_latents": 0.5],
                                     nsteps: 20, fullNsteps: 60,
                                     modalities: Self.modalities)
        }
        #expect(ScheduleError.partialTimeMismatch(dataMode: "bb_ca").description
                == "bb_ca: partial corruption time must match the first time of the full schedule tail.")
    }

    // MARK: test_fraction_rejects_invalid_step_count

    @Test("an out-of-range partial step count is rejected", arguments: [0, 61])
    func rejectsInvalidStepCount(remainingSteps: Int) {
        #expect(throws: ScheduleError.partialStepCountOutOfRange(nsteps: remainingSteps,
                                                                 fullNsteps: 60)) {
            try buildPartialSchedule(partialT: [:], nsteps: remainingSteps,
                                     fullNsteps: 60, modalities: Self.modalities)
        }
    }

    // MARK: schedule builders (not upstream tests; they pin the shapes those rely on)

    @Test("every schedule spans [~0, 1] and increases", arguments: ScheduleMode.allCases)
    func schedulesAreMonotonic(mode: ScheduleMode) throws {
        let t = try getSchedule(mode: mode, nsteps: 60, p1: 2.0)
        #expect(t.count == 61)
        #expect(t[0] == 0.0)
        #expect(abs(t[60] - 1.0) < 1e-6)
        #expect(zip(t, t.dropFirst()).allSatisfy { $0 < $1 })
    }

    @Test("TSR-safe schedules never start at zero", arguments: ScheduleMode.allCases)
    func tsrSafeSchedulesAvoidZero(mode: ScheduleMode) throws {
        // score_to_vf divides by t, so t=0 is not representable.
        let t = try getScheduleTSRSafe(mode: mode, nsteps: 60, p1: 2.0)
        #expect(t.count == 61)
        #expect(t[0] > 0, "\(mode.rawValue): TSR-safe schedule must start above 0")
        #expect(abs(t[60] - 1.0) < 1e-6)
        #expect(zip(t, t.dropFirst()).allSatisfy { $0 < $1 })
    }

    @Test("the tail multiplier keeps the step count and densifies the end")
    func tailMultiplierDensifiesTheTail() throws {
        let plain = try getScheduleTSRSafe(mode: .log, nsteps: 60, p1: 2.0)
        let tailed = try getScheduleTSRSafe(mode: .log, nsteps: 60, p1: 2.0,
                                            tailSteps: 10, tailStepMultiplier: 2)
        #expect(tailed.count == plain.count, "the tail multiplier must not change nsteps")

        // The last 10 intervals should be finer than the equivalent plain ones.
        func lastGap(_ t: [Float]) -> Float { t[t.count - 1] - t[t.count - 2] }
        #expect(lastGap(tailed) < lastGap(plain))
    }

    @Test("tail multiplier arguments are validated")
    func tailMultiplierValidation() {
        #expect(throws: ScheduleError.tailStepsNotPositive) {
            try scheduleWithTailMultiplier(nsteps: 60, tailSteps: 0, multiplier: 2) { _ in [] }
        }
        #expect(throws: ScheduleError.tailMultiplierNotInteger) {
            try scheduleWithTailMultiplier(nsteps: 60, tailSteps: 10, multiplier: 1) { _ in [] }
        }
        #expect(throws: ScheduleError.nstepsTooSmallForTail) {
            // base_steps = 10 - 1*10 = 0 < tail_steps
            try scheduleWithTailMultiplier(nsteps: 10, tailSteps: 10, multiplier: 2) { _ in [] }
        }
    }

    @Test("ode gt is identically zero regardless of its parameters")
    func odeGtIsZero() {
        let t = linspace(0, 1, 61)
        #expect(getGt(t: t, mode: .ode, param: 3.0, clampVal: 7.0).allSatisfy { $0 == 0 })
    }

    @Test("the released schedules build at 60 steps")
    func releasedSchedulesBuild() throws {
        let s = try sampleSchedule(nsteps: 60, modalities: ModalityConfig.released)
        for dm in Self.dataModes {
            #expect(s.ts[dm]?.count == 61)
            #expect(s.gt[dm]?.count == 60)
            #expect(s.gt[dm]?.allSatisfy { $0.isFinite } == true)
            #expect(s.ts[dm]?.allSatisfy { $0.isFinite } == true)
        }
    }
}
