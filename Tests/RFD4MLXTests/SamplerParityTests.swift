import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// Parity for the integration step, across every branch it can take.
///
/// `simulation_step` branches on `t`: deterministic above `t_lim_ode`, deterministic
/// below `t_lim_ode_below`, stochastic in between. A fixture at a single `t` would
/// exercise one third of the code, so these sweep all three regions for both
/// modalities — which carry different TSR constants (k=16, σ=3 for the backbone;
/// k=4, σ=12 for the latents) and so take different paths through the rescaling.
///
/// Noise is supplied from the fixture rather than drawn, so the stochastic branch is
/// a real comparison instead of two different samples.
@Suite("sampler step parity")
struct SamplerParityTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable { let shape: [Int]; let b64: String }
        struct Case: Decodable, Sendable {
            let mode: String
            let t: Float, dt: Float, gt: Float
            let tsrK: Float, tsrSigma: Float
            let xT: Entry, v: Entry, noise: Entry, emit: Entry, xNext: Entry
            enum CodingKeys: String, CodingKey {
                case mode, t, dt, gt, v, noise, emit
                case tsrK = "tsr_k"
                case tsrSigma = "tsr_sigma"
                case xT = "x_t"
                case xNext = "x_next"
            }
        }
        let torch: String
        let cases: [Case]
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "steps", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func array(_ e: Fixture.Entry) -> MLXArray {
        let data = Data(base64Encoded: e.b64)!
        return MLXArray(data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    @Test("every branch of the integration step matches upstream")
    func stepParity() throws {
        let fx = try #require(Self.fixture, "steps.json missing — run dump_step_fixtures.py")
        #expect(fx.cases.count == 6, "expected both modalities across three t regions")

        for c in fx.cases {
            let got = Sampler.step(
                xT: Self.array(c.xT), v: Self.array(c.v),
                t: c.t, dt: c.dt, gt: c.gt,
                params: Sampler.StepParams(tsrK: c.tsrK, tsrSigma: c.tsrSigma),
                emitMask: Self.array(c.emit),
                noise: Self.array(c.noise)
            )
            let want = Self.array(c.xNext)
            let d = MLX.abs(got - want)
            let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
            let rel = d.max().item(Float.self) / refMax
            let label = "\(c.mode) t=\(c.t)"
            let detail = String(format: "max |Δ| %.3e  max rel %.3e", d.max().item(Float.self), rel)
            #expect(rel <= 1e-5, "\(label): \(detail)")
            if rel <= 1e-5 { print("ok   step \(label): \(detail)") }
        }
    }

    @Test("score and vector field convert back to each other")
    func scoreRoundTrip() {
        // vf_to_score and score_to_vf are inverses; a sign or (1-t) slip in either
        // would still produce plausible trajectories, so pin the identity.
        let x = MLXArray((0 ..< 24).map { Float($0) * 0.1 - 1.2 }, [1, 8, 3])
        let v = MLXArray((0 ..< 24).map { Float($0) * -0.07 + 0.4 }, [1, 8, 3])
        for t in [Float(0.01), 0.3, 0.77, 0.99] {
            let back = Sampler.scoreToVF(xT: x, score: Sampler.vfToScore(xT: x, v: v, t: t), t: t)
            let rel = MLX.abs(back - v).max().item(Float.self)
                    / max(MLX.abs(v).max().item(Float.self), 1e-12)
            #expect(rel < 1e-5, "round trip failed at t=\(t), rel \(rel)")
        }
    }
}
