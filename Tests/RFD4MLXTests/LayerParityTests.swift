import Foundation
import MLX
import Testing
@testable import RFD4MLX

/// Module-boundary parity for the flow trunk, against activations captured from
/// upstream PyTorch on CPU.
///
/// WHY THIS NEEDS AN ENV VAR. The checkpoint is 1.74 GB and is not in the
/// repository, so these tests read it from `RFD4_WEIGHTS` and skip when it is
/// unset. The fixtures (inputs and expected outputs) are committed; only the
/// weights are not. That keeps the suite runnable in CI while leaving real parity
/// available on any machine with the checkpoint.
///
///     RFD4_WEIGHTS=~/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/\
///       snapshots/<rev>/model.safetensors swift test
@Suite("trunk layer parity")
struct LayerParityTests {

    // MARK: fixture

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable {
            let shape: [Int]?
            let b64: String?
            let ref: String?
        }
        struct Case: Decodable, Sendable {
            let inputs: [String: Entry]
            let output: Entry?
        }
        let torch: String
        let nTokens: Int
        let heads: Int
        let dimHead: Int
        let shared: [String: Entry]
        let cases: [String: Case]

        enum CodingKeys: String, CodingKey {
            case torch, shared, cases
            case nTokens = "n_tokens"
            case heads
            case dimHead = "dim_head"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "layer0", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static var weightsPath: String? {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"],
              !p.isEmpty else { return nil }
        return (p as NSString).expandingTildeInPath
    }

    // MLXArray is not Sendable; the dictionary is built once and only read.
    nonisolated(unsafe) static let weights: Weights? = {
        guard let p = weightsPath else { return nil }
        return try? Weights.load(URL(fileURLWithPath: p))
    }()

    /// Decode a fixture entry into an MLXArray, following a `ref` into `shared`.
    static func array(_ e: Fixture.Entry, in fx: Fixture) throws -> MLXArray {
        var entry = e
        if let ref = e.ref {
            guard let target = fx.shared[ref] else {
                throw ParityError.danglingRef(ref)
            }
            entry = target
        }
        guard let b64 = entry.b64, let shape = entry.shape,
              let data = Data(base64Encoded: b64) else {
            throw ParityError.badEntry
        }
        let floats = data.withUnsafeBytes { raw -> [Float] in
            Array(raw.bindMemory(to: Float.self))
        }
        return MLXArray(floats, shape)
    }

    enum ParityError: Error { case danglingRef(String), badEntry, noCase(String) }

    // MARK: comparison

    struct Delta {
        let maxAbs: Float, meanAbs: Float, maxRel: Float, refMaxAbs: Float
        var summary: String {
            String(format: "max |Δ| %.3e  mean |Δ| %.3e  max rel %.3e  (|ref| max %.3e)",
                   maxAbs, meanAbs, maxRel, refMaxAbs)
        }
    }

    static func compare(_ got: MLXArray, _ want: MLXArray) -> Delta {
        let d = MLX.abs(got - want)
        let refAbs = MLX.abs(want)
        let refMax = refAbs.max().item(Float.self)
        // Relative to the tensor's own scale, not per element: a near-zero element's
        // relative error is meaningless and would dominate the figure.
        let scale = max(refMax, 1e-12)
        return Delta(
            maxAbs: d.max().item(Float.self),
            meanAbs: d.mean().item(Float.self),
            maxRel: d.max().item(Float.self) / scale,
            refMaxAbs: refMax
        )
    }

    /// fp32 against fp32 through ~20 matmuls: a few ULP of accumulated reassociation
    /// is expected, a transposed weight or a wrong norm is not. 2e-4 relative sits far
    /// below anything structural and far above float noise at this depth.
    static let tolerance: Float = 2e-4

    private func run(
        _ name: String,
        _ body: (Fixture, Weights, [String: MLXArray]) throws -> MLXArray
    ) throws {
        guard let fx = Self.fixture else {
            Issue.record("layer0.json fixture missing — regenerate with dump_layer_fixtures.py")
            return
        }
        guard let w = Self.weights else {
            // Not a failure: the checkpoint is intentionally absent from the repo.
            print("skip \(name): set RFD4_WEIGHTS to the checkpoint to run parity")
            return
        }
        guard let c = fx.cases[name], let outEntry = c.output else {
            throw ParityError.noCase(name)
        }

        var inputs = [String: MLXArray]()
        for (k, v) in c.inputs { inputs[k] = try Self.array(v, in: fx) }

        let got = try body(fx, w, inputs)
        let want = try Self.array(outEntry, in: fx)

        #expect(got.shape == want.shape, "\(name): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }

        let d = Self.compare(got, want)
        #expect(d.maxRel <= Self.tolerance, "\(name): \(d.summary)")
        if d.maxRel <= Self.tolerance { print("ok   \(name): \(d.summary)") }
    }

    // MARK: cases

    @Test("pair-bias attention block matches upstream")
    func mhbaParity() throws {
        try run("transformer_layers.0.mhba") { fx, w, i in
            let block = try PairBiasAttentionADALN(
                w.scoped("nn.transformer_layers.0.mhba"),
                heads: fx.heads, dimHead: fx.dimHead, kind: .rms
            )
            return block(i["x"]!, pairRep: i["pair_rep"]!, cond: i["cond"]!, mask: i["mask"]!)
        }
    }

    @Test("SwiGLU transition block matches upstream")
    func transitionParity() throws {
        try run("transformer_layers.0.transition") { _, w, i in
            let block = try TransitionADALN(w.scoped("nn.transformer_layers.0.transition"),
                                            kind: .rms)
            return block(i["x"]!, cond: i["cond"]!, mask: i["mask"]!)
        }
    }

    @Test("the whole trunk layer matches upstream")
    func layerParity() throws {
        try run("transformer_layers.0") { fx, w, i in
            let layer = try TrunkLayer(
                w.scoped("nn.transformer_layers.0"),
                heads: fx.heads, dimHead: fx.dimHead, kind: .rms
            )
            return layer(i["x"]!, pairRep: i["pair_rep"]!, cond: i["cond"]!, mask: i["mask"]!)
        }
    }

    // MARK: fixture integrity (runs without the checkpoint)

    @Test("the fixture carries what the parity tests need")
    func fixtureIntegrity() throws {
        let fx = try #require(Self.fixture, "layer0.json must be present and decodable")
        #expect(fx.heads == 12)
        #expect(fx.dimHead == 64)
        #expect(fx.nTokens == 8)
        for name in ["transformer_layers.0.mhba",
                     "transformer_layers.0.transition",
                     "transformer_layers.0"] {
            let c = try #require(fx.cases[name], "missing case \(name)")
            #expect(c.output != nil, "\(name) has no captured output")
            #expect(!c.inputs.isEmpty, "\(name) has no captured inputs")
        }
        // Every ref resolves — a dangling one would silently skip a test.
        for (_, c) in fx.cases {
            for (_, e) in c.inputs where e.ref != nil {
                #expect(fx.shared[e.ref!] != nil, "dangling shared ref \(e.ref!)")
            }
        }
    }
}

extension LayerParityTests {
    @Test("pair representation update matches upstream")
    func pairUpdateParity() throws {
        try run("pair_update_layers.0") { _, w, i in
            let pu = try PairReprUpdate(w.scoped("nn.pair_update_layers.0"))
            // Hook captured positional args: (x, pair_rep, mask).
            return pu(i["arg0"]!, pairRep: i["arg1"]!, mask: i["arg2"]!)
        }
    }
}
