import Foundation
import MLX
import Testing
@testable import RFD4MLX

/// Whole-trunk parity: 14 layers, 7 pair updates, recycling and all four heads,
/// against upstream's own `_run_trunk` boundary.
///
/// This is the test the per-module ones cannot replace. Each module being right in
/// isolation does not establish that they are wired together right — the pair update
/// could run after the wrong layers, the recycle projections could be swapped, a head
/// could read the wrong representation. All of those pass module parity and fail here.
@Suite("whole trunk parity")
struct TrunkParityTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable { let shape: [Int]; let b64: String }
        let torch: String
        let nTokens: Int
        let nRecycle: Int
        let inputs: [String: Entry]
        let trunkOut: [String: Entry]
        let emit: [String: Entry]
        let heads: [String: Entry]

        enum CodingKeys: String, CodingKey {
            case torch, inputs, emit, heads
            case nTokens = "n_tokens"
            case nRecycle = "n_recycle"
            case trunkOut = "trunk_out"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "trunkio", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func array(_ e: Fixture.Entry) -> MLXArray {
        let data = Data(base64Encoded: e.b64)!
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return MLXArray(floats, e.shape)
    }

    /// Looser than the per-module budget, and deliberately so: this accumulates
    /// through 14 layers, 7 pair updates and a head, so reassociation differences
    /// compound. Still three orders under the int8 floor the port targets.
    static let tolerance: Float = 2e-3

    private func check(_ got: MLXArray, _ want: MLXArray, _ label: String,
                       tol: Float? = nil) {
        let tolerance = tol ?? Self.tolerance
        #expect(got.shape == want.shape, "\(label): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }
        let d = MLX.abs(got - want)
        let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
        let maxAbs = d.max().item(Float.self)
        let rel = maxAbs / refMax
        let detail = String(
            format: "max |Δ| %.3e  mean |Δ| %.3e  max rel %.3e  (|ref| max %.3e)",
            maxAbs, d.mean().item(Float.self), rel, refMax)
        #expect(rel <= tolerance, "\(label): \(detail)")
        if rel <= tolerance { print("ok   \(label): \(detail)") }
    }

    @Test("the full trunk and its heads match upstream")
    func trunkParity() throws {
        guard let fx = Self.fixture else {
            Issue.record("trunkio.json missing — regenerate with dump_trunk_io.py")
            return
        }
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip whole-trunk parity: set RFD4_WEIGHTS to the checkpoint")
            return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let trunk = try Trunk(w)

        let sInit = Self.array(fx.inputs["s_init"]!)
        let zInit = Self.array(fx.inputs["z_init"]!)
        let cond = Self.array(fx.inputs["cond"]!)
        let mask = Self.array(fx.inputs["mask"]!)

        let out = trunk(
            sInit: sInit, zInit: zInit, cond: cond, mask: mask,
            emitBBCA: Self.array(fx.emit["bb_ca"]!),
            emitLocalLatents: Self.array(fx.emit["local_latents"]!),
            nRecycle: fx.nRecycle
        )

        // The representations first: a divergence here explains every head at once.
        check(out.s, Self.array(fx.trunkOut["s"]!), "trunk s")
        check(out.z, Self.array(fx.trunkOut["z"]!), "trunk z")

        check(out.bbCA, Self.array(fx.heads["bb_ca.v"]!), "head bb_ca.v")
        check(out.localLatents, Self.array(fx.heads["local_latents.v"]!), "head local_latents.v")
        check(try #require(out.sequenceLogits),
              Self.array(fx.heads["sequence.logits"]!), "head sequence.logits")
        check(out.insertionLogits,
              Self.array(fx.heads["insertion.logits"]!), "head insertion.logits")
    }

    /// The recycle loop above depth 1.
    ///
    /// The main fixture runs at `n_recycle: 1`, where the loop body executes once
    /// and the feedback projections see only their zero initialisation — so neither
    /// the loop nor the projections are actually exercised by it. Production runs
    /// use 120 to 300 trunk passes over 60 steps, so depth is the normal case, not
    /// an edge one.
    @Test("the trunk matches upstream at recycle depth 4")
    func recycleDepthParity() throws {
        guard let url = Bundle.module.url(forResource: "trunkio_r4", withExtension: "json"),
              let data = try? Data(contentsOf: url) else {
            Issue.record("trunkio_r4.json missing — dump_trunk_io.py <snap> <out> 8 4")
            return
        }
        let fx = try JSONDecoder().decode(Fixture.self, from: data)
        #expect(fx.nRecycle == 4, "this fixture must exercise depth > 1")
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip recycle-depth parity: set RFD4_WEIGHTS"); return
        }
        let trunk = try Trunk(Weights.load(
            URL(fileURLWithPath: (p as NSString).expandingTildeInPath)))

        let out = trunk(
            sInit: Self.array(fx.inputs["s_init"]!), zInit: Self.array(fx.inputs["z_init"]!),
            cond: Self.array(fx.inputs["cond"]!), mask: Self.array(fx.inputs["mask"]!),
            emitBBCA: Self.array(fx.emit["bb_ca"]!),
            emitLocalLatents: Self.array(fx.emit["local_latents"]!),
            nRecycle: fx.nRecycle)

        // Depth 4 runs 56 layers instead of 14 and feeds s and z back through the
        // cycle projections three times, so reassociation error accumulates and is
        // amplified by the feedback. Measured: s 2.4e-04, v 8.2e-04, z 2.2e-03,
        // against 2.1e-06 / 8.8e-07 / 3.6e-06 at depth 1. The pair representation
        // compounds hardest because it is the thing being fed back.
        //
        // Still three orders under the int8 floor (3e-4 is the TARGET there, and
        // these are fp32-vs-fp32), and the shape of the growth is accumulation,
        // not a structural error, which a wrong feedback projection would show as
        // a large constant offset at every depth.
        let depthTolerance: Float = 5e-3
        check(out.s, Self.array(fx.trunkOut["s"]!), "depth4 trunk s", tol: depthTolerance)
        check(out.z, Self.array(fx.trunkOut["z"]!), "depth4 trunk z", tol: depthTolerance)
        check(out.bbCA, Self.array(fx.heads["bb_ca.v"]!), "depth4 head bb_ca.v",
              tol: depthTolerance)
    }

    @Test("the trunk is wired as 14 layers with pair updates only at even slots")
    func trunkShape() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let trunk = try Trunk(w)
        #expect(trunk.layers.count == 14)
        #expect(trunk.pairUpdates.count == 13)
        for (i, pu) in trunk.pairUpdates.enumerated() {
            #expect((pu != nil) == (i % 2 == 0),
                    "pair update slot \(i) should be \(i % 2 == 0 ? "populated" : "empty")")
        }
        #expect(trunk.pairUpdates.compactMap { $0 }.count == 7)
    }
}
