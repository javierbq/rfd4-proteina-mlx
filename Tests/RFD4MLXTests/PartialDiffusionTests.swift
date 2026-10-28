import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// Partial diffusion: start from a structure, corrupt it to a chosen time, denoise
/// the remaining tail — checked against upstream's own run, driven by its own
/// randomness.
///
/// Four things have to agree and only the last is shared with ordinary generation:
/// the clean state (which needs the encoder), the restart time, the corruption, and
/// the tail schedule. The corruption is where this flow can go wrong invisibly —
/// get `x_0` wrong and the result is still Gaussian-looking noise of about the right
/// size, with only the distribution off, which no single sample reveals.
@Suite("partial diffusion")
struct PartialDiffusionTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable { let shape: [Int]; let b64: String? }
        struct Mode: Decodable, Sendable {
            let dim: Int
            let zeroCOMNoise: Bool
            let stochasticCenteringMode: String
            let stochasticCenteringScale: Float?
            let groupKeys: [String]
            enum CodingKeys: String, CodingKey {
                case dim
                case zeroCOMNoise = "zero_com_noise"
                case stochasticCenteringMode = "stochastic_centering_mode"
                case stochasticCenteringScale = "stochastic_centering_scale"
                case groupKeys = "group_keys"
            }
        }
        struct Step: Decodable, Sendable {
            let step: Int
            let draws: [Entry]
            let xNext: [String: Entry]
            enum CodingKeys: String, CodingKey {
                case step, draws
                case xNext = "x_next"
            }
        }
        let nTokens: Int
        let fullNsteps: Int
        let fraction: Double
        let restartIndex: Int
        let nsteps: Int
        let partialT: [String: Float]
        let settings: [String: Mode]
        let fields: [String: Entry]
        let ts: [String: Entry]
        let gt: [String: Entry]
        let x1: [String: Entry]
        let x0: [String: Entry]
        let xt: [String: Entry]
        let corruptionDraws: [Entry]
        let steps: [Step]
        let xFinal: [String: Entry]
        enum CodingKeys: String, CodingKey {
            case fraction, settings, fields, ts, gt, steps
            case nTokens = "n_tokens"
            case fullNsteps = "full_nsteps"
            case restartIndex = "restart_index"
            case nsteps
            case partialT = "partial_t"
            case x1 = "x_1"
            case x0 = "x_0"
            case xt = "x_t"
            case corruptionDraws = "corruption_draws"
            case xFinal = "x_final"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "partialdiffusion",
                                          withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func array(_ e: Fixture.Entry) -> MLXArray? {
        guard let b64 = e.b64, let d = Data(base64Encoded: b64) else { return nil }
        return MLXArray(d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    static func entry(_ e: Fixture.Entry?, _ what: String) throws -> MLXArray {
        let e = try #require(e, "missing \(what)")
        return try #require(array(e), "\(what) was summarised, not stored")
    }

    static func field(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture, "partialdiffusion.json missing")
        return try entry(fx.fields[k], "field \(k)")
    }

    private func check(_ got: MLXArray, _ want: MLXArray, _ label: String,
                       tol: Float = 1e-4) {
        #expect(got.shape == want.shape, "\(label): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }
        let d = MLX.abs(got - want)
        let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
        let rel = d.max().item(Float.self) / refMax
        let detail = String(format: "%@: max |Δ| %.3e  max rel %.3e", label,
                            d.max().item(Float.self), rel)
        #expect(rel <= tol, "\(detail)")
        if rel <= tol { print("ok   partial \(detail)") }
    }

    // MARK: - the schedule arithmetic

    /// A fraction names a step count and the time is read off the schedule at that
    /// index — it is not computed from the fraction.
    @Test("the restart index and corruption time match upstream's")
    func restartPoint() throws {
        let fx = try #require(Self.fixture, "partialdiffusion.json missing")
        #expect(try PartialDiffusion.remainingSteps(fraction: fx.fraction,
                                                    nsteps: fx.fullNsteps) == fx.nsteps)
        #expect(try PartialDiffusion.restartIndex(fraction: fx.fraction,
                                                  nsteps: fx.fullNsteps) == fx.restartIndex)

        let full = try sampleSchedule(nsteps: fx.fullNsteps,
                                      modalities: ModalityConfig.released)
        let t = try PartialDiffusion.partialT(fraction: fx.fraction,
                                              nsteps: fx.fullNsteps, schedule: full)
        for (mode, want) in fx.partialT {
            let got = try #require(t[mode], "no partial_t for \(mode)")
            #expect(abs(got - want) < 1e-6,
                    "\(mode): restart at \(got), upstream \(want)")
        }
        // The two modalities restart at very different times from the same fraction —
        // 0.97 for the backbone against 0.50 for the sequence — because they have
        // different schedules. A port that derived one time and shared it would
        // corrupt the sequence far too little.
        let bb = try #require(t["bb_ca"]), ll = try #require(t["local_latents"])
        #expect(abs(bb - ll) > 0.3, "the two modalities no longer differ; check the schedules")
    }

    @Test("the tail is the last nsteps intervals of the full schedule")
    func tailSchedule() throws {
        let fx = try #require(Self.fixture, "partialdiffusion.json missing")
        let full = try sampleSchedule(nsteps: fx.fullNsteps,
                                      modalities: ModalityConfig.released)
        let tail = try PartialDiffusion.tail(of: full, nsteps: fx.nsteps)
        for mode in ["bb_ca", "local_latents"] {
            let ts = try #require(tail.ts[mode]), gt = try #require(tail.gt[mode])
            // ts keeps nsteps + 1 endpoints, gt keeps nsteps intervals.
            #expect(ts.count == fx.nsteps + 1)
            #expect(gt.count == fx.nsteps)
            check(MLXArray(ts), try Self.entry(fx.ts[mode], "ts[\(mode)]"), "ts[\(mode)]")
            check(MLXArray(gt), try Self.entry(fx.gt[mode], "gt[\(mode)]"), "gt[\(mode)]")
            // The tail must begin exactly at the corruption time; upstream asserts it.
            #expect(abs(ts[0] - fx.partialT[mode]!) < 1e-6)
        }
    }

    /// A larger fraction must restart earlier, in both modalities.
    ///
    /// This is the knob's whole contract — "how much do I trust the model to change
    /// what I gave it" — and it is the half of it that costs nothing to check. The
    /// other half needs the weights: measured on a 225-token complex, CA drift from
    /// the input goes 0.30 Å, 0.71 Å, 2.20 Å, 8.53 Å at fractions 0.3, 0.5, 0.7, 0.9.
    @Test("a larger fraction restarts earlier and runs longer")
    func fractionIsMonotonic() throws {
        let full = try sampleSchedule(nsteps: 60, modalities: ModalityConfig.released)
        var previous: [String: Float] = ["bb_ca": 2, "local_latents": 2]
        var previousSteps = 0
        for fraction in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let steps = try PartialDiffusion.remainingSteps(fraction: fraction, nsteps: 60)
            #expect(steps > previousSteps, "fraction \(fraction) did not run longer")
            previousSteps = steps
            let t = try PartialDiffusion.partialT(fraction: fraction, nsteps: 60,
                                                  schedule: full)
            for (mode, value) in t {
                let why = "\(mode) at fraction \(fraction) restarts at \(value), not earlier than \(previous[mode]!)"
                #expect(value < previous[mode]!, "\(why)")
                previous[mode] = value
            }
        }
    }

    @Test("invalid restarts are refused")
    func restartErrors() throws {
        #expect(throws: PartialDiffusionError.self) {
            try PartialDiffusion.remainingSteps(fraction: 0, nsteps: 60)
        }
        #expect(throws: PartialDiffusionError.self) {
            try PartialDiffusion.remainingSteps(fraction: 1.5, nsteps: 60)
        }
        // Rounds to zero steps: a run that denoises nothing returns the corrupted
        // input and looks like a very conservative design.
        #expect(throws: PartialDiffusionError.self) {
            try PartialDiffusion.remainingSteps(fraction: 0.001, nsteps: 60)
        }
        let full = try sampleSchedule(nsteps: 60, modalities: ModalityConfig.released)
        #expect(throws: PartialDiffusionError.self) {
            try PartialDiffusion.tail(of: full, nsteps: 61)
        }
        #expect(throws: PartialDiffusionError.self) {
            try PartialDiffusion.tail(of: full, nsteps: 0)
        }
    }

    // MARK: - the corruption

    /// The settings the corruption depends on, as the checkpoint carries them.
    @Test("the released corruption settings are what the port assumes")
    func corruptionSettings() throws {
        let fx = try #require(Self.fixture, "partialdiffusion.json missing")
        let bb = try #require(fx.settings["bb_ca"])
        let ll = try #require(fx.settings["local_latents"])
        #expect(bb.zeroCOMNoise && bb.stochasticCenteringMode == "x_0"
                && bb.stochasticCenteringScale == 0.2 && bb.groupKeys == ["molecule_iid"])
        #expect(!ll.zeroCOMNoise && ll.stochasticCenteringMode == "none")

        let released = Corruption.Settings.released
        #expect(released["bb_ca"]?.zeroCOMNoise == bb.zeroCOMNoise)
        #expect(released["bb_ca"]?.stochasticCenteringScale == bb.stochasticCenteringScale)
        #expect(released["local_latents"]?.zeroCOMNoise == ll.zeroCOMNoise)
        #expect(released["local_latents"]?.stochasticCenteringScale == nil)
    }

    @Test("the corruption reproduces upstream's, draw for draw")
    func corruptionParity() throws {
        let fx = try #require(Self.fixture, "partialdiffusion.json missing")
        let padding = try Self.field("padding_mask")
        let context = try Self.field("is_in_context_conditioning")
        let supports = try Self.field("supports_local_latent")
        let emitBB = padding * (1 - context)
        let emitMasks = ["bb_ca": emitBB, "local_latents": emitBB * supports]

        // Upstream's draws, in the order it made them: both base noises, then the
        // per-group offsets. Four for a two-chain complex.
        #expect(fx.corruptionDraws.count == 4,
                "expected 4 draws (2 base + 2 molecule offsets), got \(fx.corruptionDraws.count)")
        var base = [String: MLXArray]()
        base["bb_ca"] = try Self.entry(fx.corruptionDraws[0], "base bb_ca")
        base["local_latents"] = try Self.entry(fx.corruptionDraws[1], "base local_latents")
        var offsets: [MLXArray] = []
        for i in 2 ..< fx.corruptionDraws.count {
            offsets.append(try Self.entry(fx.corruptionDraws[i], "offset \(i)"))
        }
        var consumed = 0

        let result = Corruption.corrupt(
            x1: ["bb_ca": try Self.entry(fx.x1["bb_ca"], "x_1 bb_ca"),
                 "local_latents": try Self.entry(fx.x1["local_latents"], "x_1 latents")],
            t: fx.partialT,
            emitMasks: emitMasks,
            groupIDs: try Self.field("molecule_iid"),
            baseNoise: { mode, _ in base[mode]! },
            offsetDraw: { _, _, _ in
                defer { consumed += 1 }
                return offsets[min(consumed, offsets.count - 1)]
            })

        #expect(consumed == offsets.count,
                "drew \(consumed) group offsets, upstream drew \(offsets.count)")
        for mode in ["bb_ca", "local_latents"] {
            // `x_0` as upstream records it: before the per-group translations, which
            // it applies inside `interpolate` to a local variable.
            check(result.x0[mode]!, try Self.entry(fx.x0[mode], "x_0 \(mode)"), "x_0 \(mode)")
            check(result.xt[mode]!, try Self.entry(fx.xt[mode], "x_t \(mode)"), "x_t \(mode)")
        }

        // And the consequence, stated as a fact rather than left implicit: for the
        // backbone the recorded `x_0` does NOT reconstruct `x_t`, because the
        // translations are not in it. For the latents, which have no centering, it
        // does. A port that conflated the two would pass on the latents and be
        // quietly wrong on the backbone.
        let tBB = fx.partialT["bb_ca"]!
        let naive = Corruption.interpolate(
            x0: result.x0["bb_ca"]!, x1: try Self.entry(fx.x1["bb_ca"], "x_1 bb_ca"),
            t: tBB, emitMask: emitBB)
        let gap = MLX.abs(naive - result.xt["bb_ca"]!).max().item(Float.self)
        #expect(gap > 1e-4,
                "the recorded x_0 now reconstructs x_t; the offsets have gone missing")
        #expect(MLX.abs(result.x0Interpolated["local_latents"]!
                        - result.x0["local_latents"]!).max().item(Float.self) == 0,
                "local_latents was translated; its centering mode is none")
    }

    /// The per-group translation is per group, and it is not nothing.
    @Test("each molecule is translated independently")
    func stochasticOffsetIsPerGroup() throws {
        let groups = MLXArray([Float](repeating: 0, count: 4)
                              + [Float](repeating: 1, count: 4)).reshaped(1, 8)
        let x = MLX.zeros([1, 8, 3])
        let mask = MLX.ones([1, 8])
        var seen: [Int] = []
        let out = Corruption.stochasticOffset(
            x, groupIDs: groups, offsetMask: mask, scale: 0.2,
            draw: { group, dim in
                seen.append(group)
                return MLXArray([Float](repeating: Float(group) + 1, count: dim))
            })
        // One draw per group, in ascending order.
        #expect(seen == [0, 1])
        let v = out[0].asArray(Float.self)
        #expect(v[0] == 0.2 && v[1] == 0.2 && v[2] == 0.2, "group 0 offset wrong")
        #expect(v[12] == 0.4 && v[13] == 0.4, "group 1 got group 0's offset")

        // A group with no eligible token must not consume a draw: upstream iterates
        // `group_ids[offset_mask].unique()`, so drawing for an absent group would
        // shift every later draw in a matched-noise replay.
        var half = [Float](repeating: 1, count: 8)
        for i in 4 ..< 8 { half[i] = 0 }
        seen = []
        _ = Corruption.stochasticOffset(
            x, groupIDs: groups, offsetMask: MLXArray(half).reshaped(1, 8), scale: 0.2,
            draw: { group, dim in
                seen.append(group)
                return MLX.zeros([dim])
            })
        #expect(seen == [0], "drew for a group with no eligible token")
    }

    // MARK: - the whole trajectory

    @Test("the partial trajectory follows upstream's own")
    func trajectoryParity() throws {
        let fx = try #require(Self.fixture, "partialdiffusion.json missing")
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip partial trajectory: set RFD4_WEIGHTS"); return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let featurizer = try Featurizer(w)
        let trunk = try Trunk(w)

        let padding = try Self.field("padding_mask")
        let context = try Self.field("is_in_context_conditioning")
        let supports = try Self.field("supports_local_latent")
        let emitBB = padding * (1 - context)
        let emitLL = emitBB * supports

        let tokenType = try Self.field("token_type")
        let cath = try Self.field("condition_cath-label_1_residue")
        let coordGroup = try Self.field("condition_CoordinateGroup_1_atom")
        let indexGroup = try Self.field("condition_index-group_1_token")
        let moleculeIID = try Self.field("molecule_iid")
        let moleculeEntity = try Self.field("molecule_entity_id")
        let bondOrder = try Self.field("bond_order")
        let pathLength = try Self.field("token_bond_path_length")
        let ordering = try Self.field("canonical_polymer_ordering")

        var byStep: [Int: [Int: MLXArray]] = [:]
        for s in fx.steps {
            var m: [Int: MLXArray] = [:]
            for d in s.draws { if let a = Self.array(d) { m[d.shape.last!] = a } }
            byStep[s.step] = m
        }

        let full = try sampleSchedule(nsteps: fx.fullNsteps,
                                      modalities: ModalityConfig.released)
        let tail = try PartialDiffusion.tail(of: full, nsteps: fx.nsteps)

        let x = Sampler.simulate(
            x0: ["bb_ca": try Self.entry(fx.xt["bb_ca"], "x_t bb_ca"),
                 "local_latents": try Self.entry(fx.xt["local_latents"], "x_t latents")],
            schedule: tail, nsteps: fx.nsteps,
            recycleDepths: Array(repeating: 1, count: fx.nsteps),
            params: Sampler.StepParams.released,
            emitMasks: ["bb_ca": emitBB, "local_latents": emitLL],
            predict: { state, t, depth in
                let b = Featurizer.Batch(
                    xtBBCA: state["bb_ca"]!, xtLocalLatents: state["local_latents"]!,
                    tBBCA: t["bb_ca"]!, tLocalLatents: t["local_latents"]!,
                    tokenType: tokenType, paddingMask: padding,
                    cathCode: cath, coordGroup: coordGroup,
                    indexGroup: indexGroup, moleculeIID: moleculeIID,
                    moleculeEntityID: moleculeEntity, bondOrder: bondOrder,
                    tokenBondPathLength: pathLength,
                    canonicalPolymerOrdering: ordering)
                let (sInit, zInit, cond) = featurizer(b)
                let o = trunk(sInit: sInit, zInit: zInit, cond: cond, mask: padding,
                              emitBBCA: emitBB, emitLocalLatents: emitLL, nRecycle: depth)
                return ["bb_ca": o.bbCA, "local_latents": o.localLatents]
            },
            noiseFor: { _, step, shape in byStep[step]?[shape.last!] })

        for mode in ["bb_ca", "local_latents"] {
            check(x[mode]!, try Self.entry(fx.xFinal[mode], "x_final \(mode)"),
                  "trajectory \(mode)", tol: 5e-2)
        }
    }
}
