import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// Isolates `predict` from `step` in the collapsing-chain defect.
///
/// Every single-call parity test passes, yet the Swift loop collapses while
/// upstream's does not on the same batch — so the fault lies in a sequence. These
/// drive the Swift featurizer + trunk from states captured mid-run by the reference
/// and compare the vector field it predicts. If `v` matches at every sampled time,
/// `predict` is exonerated and the fault is in how the loop is driven.
@Suite("predict-step parity")
struct PredictStepTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable { let shape: [Int]; let b64: String }
        let n: Int
        let nsteps: Int
        let emit: [String: Entry]
        let steps: [String: [String: Entry]]
        let trunkIo: [String: [String: Entry]]?
        enum CodingKeys: String, CodingKey {
            case n, nsteps, emit, steps
            case trunkIo = "trunk_io"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "predictsteps", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func arr(_ e: Fixture.Entry) -> MLXArray {
        let d = Data(base64Encoded: e.b64)!
        return MLXArray(d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    @Test("the predicted vector field matches at every sampled time")
    func predictParity() throws {
        let fx = try #require(Self.fixture, "predictsteps.json missing")
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip predict parity: set RFD4_WEIGHTS"); return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let featurizer = try Featurizer(w)
        let trunk = try Trunk(w)
        let system = Generator.System.deNovo(length: fx.n)
        let emitBB = Self.arr(try #require(fx.emit["bb_ca"]))
        let emitLL = Self.arr(try #require(fx.emit["local_latents"]))

        for key in fx.steps.keys.sorted(by: { Int($0)! < Int($1)! }) {
            let s = fx.steps[key]!
            let t = Self.arr(s["t.bb_ca"]!).item(Float.self)
            let b = Featurizer.Batch(
                xtBBCA: Self.arr(s["x_t.bb_ca"]!),
                xtLocalLatents: Self.arr(s["x_t.local_latents"]!),
                tBBCA: t,
                tLocalLatents: Self.arr(s["t.local_latents"]!).item(Float.self),
                tokenType: system.tokenType, paddingMask: system.paddingMask,
                cathCode: system.cathCode, coordGroup: system.coordGroup,
                indexGroup: system.indexGroup, moleculeIID: system.moleculeIID,
                moleculeEntityID: system.moleculeEntityID, bondOrder: system.bondOrder,
                tokenBondPathLength: system.tokenBondPathLength,
                canonicalPolymerOrdering: system.canonicalPolymerOrdering)

            let (sInit, zInit, cond) = featurizer(b)

            // Localise: is the featurization wrong, or the trunk call?
            if let io = fx.trunkIo?[key] {
                for (nm, got, want) in [
                    ("s_init", sInit, Self.arr(io["s_init"]!)),
                    ("z_init", zInit, Self.arr(io["z_init"]!)),
                    ("cond", cond, Self.arr(io["cond"]!)),
                ] {
                    let d = MLX.abs(got - want)
                    let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
                    print(String(format: "     step %@ %@: max rel %.3e", key, nm,
                                 d.max().item(Float.self) / refMax))
                }
            }
            let out = trunk(sInit: sInit, zInit: zInit, cond: cond,
                            mask: system.paddingMask,
                            emitBBCA: emitBB, emitLocalLatents: emitLL, nRecycle: 1)

            for (name, got, want) in [
                ("bb_ca", out.bbCA, Self.arr(s["v.bb_ca"]!)),
                ("local_latents", out.localLatents, Self.arr(s["v.local_latents"]!)),
            ] {
                let d = MLX.abs(got - want)
                let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
                let rel = d.max().item(Float.self) / refMax
                let detail = String(format: "t=%.5f  max |Δ| %.3e  max rel %.3e  "
                                    + "(|ref| max %.3e)", t,
                                    d.max().item(Float.self), rel, refMax)
                #expect(rel <= 2e-3, "step \(key) v.\(name): \(detail)")
                print("     step \(key) v.\(name): \(detail)")
            }
        }
    }
}
