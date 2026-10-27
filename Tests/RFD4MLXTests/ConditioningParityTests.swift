import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// The four feature creators that carry a target, checked against a REAL binder
/// design — upstream's shipped `protein_binder/input.cif`, 225 tokens.
///
/// The de-novo fixtures cannot test these at all: all four are identically zero
/// without a target, so a broken implementation passes every unconditioned test.
/// Measured on this input, 17 of 35 creators are live against 13 de-novo, and these
/// are the four that differ.
@Suite("conditioning parity")
struct ConditioningParityTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable {
            let shape: [Int]
            let b64: String?
            let dead: Bool
        }
        let nTokens: Int
        let source: String
        let inputs: [String: Entry]
        let features: [String: Entry]
        let seqProbKeys: [String]
        enum CodingKeys: String, CodingKey {
            case source, inputs, features
            case nTokens = "n_tokens"
            case seqProbKeys = "seq_prob_keys"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "conditioned", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func arr(_ e: Fixture.Entry) -> MLXArray? {
        guard let b64 = e.b64, let d = Data(base64Encoded: b64) else { return nil }
        return MLXArray(d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    // #require does not nest, so each lookup is its own step.
    static func input(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture, "conditioned.json missing")
        let entry = try #require(fx.inputs[k], "no input \(k)")
        return try #require(arr(entry), "input \(k) was summarised, not stored")
    }

    static func feature(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture, "conditioned.json missing")
        let entry = try #require(fx.features[k], "no feature \(k)")
        return try #require(arr(entry), "feature \(k) was summarised, not stored")
    }

    private func check(_ got: MLXArray, _ want: MLXArray, _ label: String) {
        #expect(got.shape == want.shape, "\(label): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }
        let d = MLX.abs(got - want)
        let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
        let rel = d.max().item(Float.self) / refMax
        let detail = String(format: "max |Δ| %.3e  max rel %.3e  (|ref| max %.3e)",
                            d.max().item(Float.self), rel, refMax)
        #expect(rel <= 1e-5, "\(label): \(detail)")
        if rel <= 1e-5 { print("ok   cond \(label): \(detail)") }
    }

    @Test("the fixture really is a conditioned binder design")
    func fixtureIsConditioned() throws {
        let fx = try #require(Self.fixture, "conditioned.json missing")
        #expect(fx.source.contains("protein_binder"))
        #expect(fx.nTokens == 225)
        let live = fx.features.values.filter { !$0.dead }.count
        #expect(live == 17, "expected 17 live creators for a binder design, got \(live)")
        // The four that separate conditioned from de-novo must all carry signal —
        // otherwise the tests below would compare zeros to zeros and pass.
        for key in ["init_repr_factory/ConditionalAtom37NanometersCoorsSeqFeat",
                    "init_repr_factory/SequenceProbabilitiesSeqFeat",
                    "init_repr_factory/HotspotSeqFeat",
                    "pair_repr_builder/CanonicalPolymerOrderingPairFeat"] {
            #expect(fx.features[key]?.dead == false, "\(key) is inert in this fixture")
        }
    }

    @Test("conditional atom37 coordinates")
    func conditionalCoords() throws {
        // The batch carries Angstrom with NaN for unconditioned positions.
        let coords = try Self.input("coord_cond_mask")
        let got = Features.conditionalAtom37Coords(coords)
        check(got, try Self.feature("init_repr_factory/ConditionalAtom37NanometersCoorsSeqFeat"),
              "ConditionalAtom37NanometersCoorsSeqFeat")
    }

    @Test("hotspot annotation")
    func hotspots() throws {
        let got = Features.hotspot(try Self.input("hotspot"),
                                   atom37Mask: try Self.input("atom37_mask"))
        check(got, try Self.feature("init_repr_factory/HotspotSeqFeat"), "HotspotSeqFeat")
    }

    @Test("sequence probabilities carry their specified-mask")
    func sequenceProbabilities() throws {
        let fx = try #require(Self.fixture)
        #expect(fx.seqProbKeys.count == 28, "expected 28 channels, got \(fx.seqProbKeys.count)")
        var channels: [MLXArray] = []
        for key in fx.seqProbKeys {
            channels.append(try Self.input("seqprob/" + key))
        }
        let got = Features.sequenceProbabilities(channels)
        check(got, try Self.feature("init_repr_factory/SequenceProbabilitiesSeqFeat"),
              "SequenceProbabilitiesSeqFeat")
        // The second half is the mask; it must not be all ones, or the
        // explicit-zero-vs-unconstrained distinction this feature exists for is lost.
        let k = fx.seqProbKeys.count
        let maskHalf = got[0..., 0..., k ..< (2 * k)]
        let allSpecified = MLX.all(maskHalf .> 0).item(Bool.self)
        #expect(!allSpecified, "every position reads as specified — the mask is degenerate")
    }

    @Test("canonical polymer ordering is signed")
    func polymerOrdering() throws {
        let ordering = try Self.input("canonical_polymer_ordering")
        let got = Features.canonicalPolymerOrdering(ordering)
        check(got, try Self.feature("pair_repr_builder/CanonicalPolymerOrderingPairFeat"),
              "CanonicalPolymerOrderingPairFeat")
        // +1 downstream/upstream, -1 at the transpose. If the port ever collapsed it
        // to adjacency the sign would vanish and chain direction with it.
        #expect(MLX.any(got .< 0).item(Bool.self), "no negative entries — direction lost")
        #expect(MLX.any(got .> 0).item(Bool.self), "no positive entries")
    }
}
