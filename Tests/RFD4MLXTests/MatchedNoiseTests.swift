import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// Drives the Swift sampler down upstream's EXACT trajectory.
///
/// Comparing two independent 16-residue samples cannot separate a real divergence
/// from sampling variance — which is why the earlier Rg comparisons could neither
/// confirm nor refute a defect. Starting from the reference's own initial state and
/// feeding its own Gaussian draws removes the variance entirely: any difference
/// that remains is the port's.
@Suite("matched noise")
struct MatchedNoiseTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable { let shape: [Int]; let b64: String }
        struct Step: Decodable, Sendable {
            let step: Int
            let draws: [Entry]
            let xNext: [String: Entry]
            enum CodingKeys: String, CodingKey {
                case step, draws
                case xNext = "x_next"
            }
        }
        let n: Int
        let nsteps: Int
        let depths: [Int]
        let x0: [String: Entry]
        let steps: [Step]
        let xFinal: [String: Entry]
        enum CodingKeys: String, CodingKey {
            case n, nsteps, depths, x0, steps
            case xFinal = "x_final"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "matchednoise", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func arr(_ e: Fixture.Entry) -> MLXArray {
        let d = Data(base64Encoded: e.b64)!
        return MLXArray(d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    @Test("the same trajectory, driven by upstream's own noise")
    func matchedTrajectory() throws {
        let fx = try #require(Self.fixture, "matchednoise.json missing")
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip matched noise: set RFD4_WEIGHTS"); return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let featurizer = try Featurizer(w)
        let trunk = try Trunk(w)
        let system = Generator.System.deNovo(length: fx.n)

        // Draws are identified by trailing dimension: bb_ca is 3, local_latents 32.
        // Upstream emits them in data_modes order within a step, which matches the
        // sorted order this port uses, but keying on shape is independent of that.
        var byStep: [Int: [Int: MLXArray]] = [:]
        for s in fx.steps {
            var m: [Int: MLXArray] = [:]
            for d in s.draws { m[d.shape.last!] = Self.arr(d) }
            byStep[s.step] = m
        }

        let schedule = try sampleSchedule(nsteps: fx.nsteps,
                                          modalities: ModalityConfig.released)
        let x = try Sampler.simulate(
            x0: ["bb_ca": Self.arr(fx.x0["bb_ca"]!),
                 "local_latents": Self.arr(fx.x0["local_latents"]!)],
            schedule: schedule, nsteps: fx.nsteps, recycleDepths: fx.depths,
            params: Sampler.StepParams.released,
            emitMasks: ["bb_ca": system.paddingMask, "local_latents": system.paddingMask],
            predict: { state, t, depth in
                let b = Featurizer.Batch(
                    xtBBCA: state["bb_ca"]!, xtLocalLatents: state["local_latents"]!,
                    tBBCA: t["bb_ca"]!, tLocalLatents: t["local_latents"]!,
                    tokenType: system.tokenType, paddingMask: system.paddingMask,
                    cathCode: system.cathCode, coordGroup: system.coordGroup,
                    indexGroup: system.indexGroup, moleculeIID: system.moleculeIID,
                    moleculeEntityID: system.moleculeEntityID,
                    bondOrder: system.bondOrder,
                    tokenBondPathLength: system.tokenBondPathLength,
                    canonicalPolymerOrdering: system.canonicalPolymerOrdering)
                let (sInit, zInit, cond) = featurizer(b)
                let o = trunk(sInit: sInit, zInit: zInit, cond: cond,
                              mask: system.paddingMask,
                              emitBBCA: system.paddingMask,
                              emitLocalLatents: system.paddingMask, nRecycle: depth)
                return ["bb_ca": o.bbCA, "local_latents": o.localLatents]
            },
            noiseFor: { _, step, shape in byStep[step]?[shape.last!] }
        )

        for mode in ["bb_ca", "local_latents"] {
            let got = x[mode]!, want = Self.arr(fx.xFinal[mode]!)
            let d = MLX.abs(got - want)
            let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
            let rel = d.max().item(Float.self) / refMax
            print(String(format: "     matched %@: max |Δ| %.3e  max rel %.3e  "
                         + "(|ref| max %.3e)", mode, d.max().item(Float.self),
                         rel, refMax))
            // 60 steps of compounding, each feeding the next through the network.
            #expect(rel <= TrajectoryGate.endpoint,
                    "\(mode) diverged from upstream's own trajectory")
        }
    }
}
