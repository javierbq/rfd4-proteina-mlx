import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// A whole CONDITIONED trajectory, driven by upstream's own randomness.
///
/// `BinderFeaturizerTests` proves the port builds the same trunk inputs for a
/// binder. This proves what happens over the 60 steps that follow — and the two are
/// not the same claim. On the de-novo path every single-call parity test passed
/// while the loop still collapsed; three real defects lived between the calls, and
/// each was invisible to a one-shot comparison.
///
/// Two independent samples cannot settle it either: sampling variance is far larger
/// than any plausible arithmetic difference, so an Rg comparison can neither confirm
/// nor refute a divergence. Replaying upstream's own initial state and its own
/// Gaussian draws removes the variance, and what remains is the port's.
///
/// The problem is small on purpose — `"20, /0, A17-50"`, 54 tokens — because the
/// reference run is PyTorch on CPU and the pair track is quadratic. Size is not what
/// is under test here; the conditioning is. The 225-token recipe is checked at the
/// featurizer instead.
@Suite("binder matched noise")
struct BinderMatchedNoiseTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable { let shape: [Int]; let b64: String? }
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
        let nsteps: Int
        let depths: [Int]
        let seqProbKeys: [String]
        let fields: [String: Entry]
        let x0: [String: Entry]
        let steps: [Step]
        let xFinal: [String: Entry]
        enum CodingKeys: String, CodingKey {
            case nsteps, depths, fields, x0, steps
            case nTokens = "n_tokens"
            case seqProbKeys = "seq_prob_keys"
            case xFinal = "x_final"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "bindernoise", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func arr(_ e: Fixture.Entry) -> MLXArray? {
        guard let b64 = e.b64, let d = Data(base64Encoded: b64) else { return nil }
        return MLXArray(d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    // #require does not nest, so each lookup is its own step.
    static func field(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture, "bindernoise.json missing")
        let e = try #require(fx.fields[k], "no field \(k)")
        return try #require(arr(e), "field \(k) was summarised, not stored")
    }

    static func entry(_ e: Fixture.Entry?, _ what: String) throws -> MLXArray {
        let e = try #require(e, "missing \(what)")
        return try #require(arr(e), "\(what) was summarised, not stored")
    }

    /// Upstream's own conditioning, so this isolates the trajectory from the
    /// assembly that `BinderLayoutTests` already checks.
    static func conditioning(_ fx: Fixture) throws -> Featurizer.Batch.Conditioning {
        let raw = try field("condition_CoordinateGroup_1_atom")
        let densified = CoordGroups.densified(raw.reshaped(-1).asArray(Float.self).map { Int($0) })
        var channels: [MLXArray] = []
        for name in fx.seqProbKeys {
            channels.append(try field("condition_sequence-probability-\(name)_1_residue"))
        }
        return Featurizer.Batch.Conditioning(
            internalCoords: try field("condition_InternalCoordinate_1_atom"),
            absoluteCoords: try field("condition_coordinate_1_atom"),
            coordGroups: raw,
            coordGroupsDensified: MLXArray(densified.map(Float.init), raw.shape),
            sequenceProbabilities: channels,
            hotspot: try field("condition_hotspot_1_atom"),
            atom37Mask: try field("atom37_mask"))
    }

    @Test("the fixture is a conditioned two-chain run, not a monomer")
    func fixtureIsConditioned() throws {
        let fx = try #require(Self.fixture, "bindernoise.json missing")
        #expect(fx.nTokens > 1)
        // A chain break shows as a zero path length between consecutive tokens.
        let path = try Self.field("token_bond_path_length")
        var breaks: [Int] = []
        for i in 1 ..< fx.nTokens where path[0, i, i - 1].item(Float.self) == 0 {
            breaks.append(i)
        }
        #expect(breaks.count == 1, "expected one chain break, found \(breaks)")
        // And the conditioning must carry signal, or every check below compares a
        // conditioned run to an unconditioned one and passes for the wrong reason.
        let internalCoords = try Self.field("condition_InternalCoordinate_1_atom")
        #expect(MLX.any(MLX.isFinite(internalCoords)).item(Bool.self),
                "no target geometry — this is not a binder")
        #expect(MLX.any(try Self.field("condition_hotspot_1_atom") .> 0).item(Bool.self))
    }

    @Test("the conditioned trajectory follows upstream's own")
    func matchedConditionedTrajectory() throws {
        let fx = try #require(Self.fixture, "bindernoise.json missing")
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip binder matched noise: set RFD4_WEIGHTS"); return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let featurizer = try Featurizer(w)
        let trunk = try Trunk(w)
        let conditioning = try Self.conditioning(fx)

        let padding = try Self.field("padding_mask")
        let tokenType = try Self.field("token_type")
        let cath = try Self.field("condition_cath-label_1_residue")
        let indexGroup = try Self.field("condition_index-group_1_token")
        let moleculeIID = try Self.field("molecule_iid")
        let moleculeEntity = try Self.field("molecule_entity_id")
        let bondOrder = try Self.field("bond_order")
        let pathLength = try Self.field("token_bond_path_length")
        let ordering = try Self.field("canonical_polymer_ordering")

        // Emit masks: padding & ~is_in_context_conditioning, and the local-latent
        // one also & supports_local_latent. Taken from the batch rather than assumed
        // all-ones, so a fixture where the target IS frozen would still be driven
        // correctly.
        let context = try Self.field("is_in_context_conditioning")
        let supports = try Self.field("supports_local_latent")
        let emitBB = padding * (1 - context)
        let emitLL = emitBB * supports

        // Draws are keyed by trailing dimension — bb_ca is 3, local_latents 32 — so
        // this does not depend on the order upstream happens to emit them in.
        var byStep: [Int: [Int: MLXArray]] = [:]
        for s in fx.steps {
            var m: [Int: MLXArray] = [:]
            for d in s.draws { if let a = Self.arr(d) { m[d.shape.last!] = a } }
            byStep[s.step] = m
        }

        let schedule = try sampleSchedule(nsteps: fx.nsteps,
                                          modalities: ModalityConfig.released)
        var worstPerStep: (step: Int, rel: Float) = (-1, 0)

        let x = Sampler.simulate(
            x0: ["bb_ca": try Self.entry(fx.x0["bb_ca"], "x0.bb_ca"),
                 "local_latents": try Self.entry(fx.x0["local_latents"], "x0.local_latents")],
            schedule: schedule, nsteps: fx.nsteps, recycleDepths: fx.depths,
            params: Sampler.StepParams.released,
            emitMasks: ["bb_ca": emitBB, "local_latents": emitLL],
            predict: { state, t, depth in
                let b = Featurizer.Batch(
                    xtBBCA: state["bb_ca"]!, xtLocalLatents: state["local_latents"]!,
                    tBBCA: t["bb_ca"]!, tLocalLatents: t["local_latents"]!,
                    tokenType: tokenType, paddingMask: padding,
                    cathCode: cath, coordGroup: conditioning.coordGroups,
                    indexGroup: indexGroup, moleculeIID: moleculeIID,
                    moleculeEntityID: moleculeEntity, bondOrder: bondOrder,
                    tokenBondPathLength: pathLength,
                    canonicalPolymerOrdering: ordering,
                    conditioning: conditioning)
                let (sInit, zInit, cond) = featurizer(b)
                let o = trunk(sInit: sInit, zInit: zInit, cond: cond, mask: padding,
                              emitBBCA: emitBB, emitLocalLatents: emitLL, nRecycle: depth)
                return ["bb_ca": o.bbCA, "local_latents": o.localLatents]
            },
            noiseFor: { _, step, shape in byStep[step]?[shape.last!] },
            onStep: { step, state in
                // Per-step, not just the end: a trajectory that drifts slowly is a
                // different fault from one that breaks at a particular t, and only
                // the step-by-step record separates them.
                guard let want = fx.steps.first(where: { $0.step == step }),
                      let ref = want.xNext["bb_ca"], let refArr = Self.arr(ref),
                      let got = state["bb_ca"] else { return }
                let d = MLX.abs(got - refArr).max().item(Float.self)
                let scale = max(MLX.abs(refArr).max().item(Float.self), 1e-12)
                if d / scale > worstPerStep.rel { worstPerStep = (step, d / scale) }
            }
        )

        print(String(format: "     worst single step: %d at max rel %.3e",
                     worstPerStep.step, worstPerStep.rel))
        // Asserted, not merely printed. A fault that peaks mid-trajectory and relaxes
        // by the endpoint passes an endpoint-only check: 3% at step 30 that decays to
        // 1% by step 59 is invisible at 5e-2. This is the number that localises such a
        // fault, and computing it and discarding it was worse than not computing it.
        // The bound is per precision (see TrajectoryGate.perStep): tight in fp32,
        // where the measured worst step is 4.1e-06, and the endpoint gate itself in
        // half precision, where it is 1.9e-02.
        let worstDetail = String(format: "diverged mid-trajectory at step %d "
                                 + "(max rel %.3e) even if it recovered by the end",
                                 worstPerStep.step, worstPerStep.rel)
        #expect(worstPerStep.rel <= TrajectoryGate.perStep, "\(worstDetail)")
        for mode in ["bb_ca", "local_latents"] {
            let got = x[mode]!
            let want = try Self.entry(fx.xFinal[mode], "x_final.\(mode)")
            let d = MLX.abs(got - want)
            let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
            let rel = d.max().item(Float.self) / refMax
            print(String(format: "     binder matched %@: max |Δ| %.3e  max rel %.3e  "
                         + "(|ref| max %.3e)", mode, d.max().item(Float.self), rel, refMax))
            // 60 steps of compounding, each feeding the next through the network, on
            // a conditioned two-chain system.
            #expect(rel <= TrajectoryGate.endpoint,
                    "\(mode) diverged from upstream's own trajectory")
        }
    }
}
