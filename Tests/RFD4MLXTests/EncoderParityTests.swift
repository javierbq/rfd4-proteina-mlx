import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// The autoencoder encoder, against upstream, on a real two-chain structure.
///
/// Checked creator by creator before the factory and the layers, because a factory
/// that disagrees could be any of six features and the factory output says nothing
/// about which. That bisection has already paid for itself twice in this port: a
/// sequence-probability channel list sorted instead of registry-ordered, and a
/// coordinate group passed through raw instead of densified — both invisible at the
/// factory, both obvious one creator down.
@Suite("encoder parity")
struct EncoderParityTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable {
            let shape: [Int]
            let b64: String?
            let offset: Int?
            let creator: String?
        }
        let nTokens: Int
        let seqWidth: Int
        let fields: [String: Entry]
        let creators: [String: Entry]
        let outputs: [String: Entry]
        let mean: Entry
        let logScale: Entry
        let zLatent: Entry
        let emitMask: Entry
        enum CodingKeys: String, CodingKey {
            case fields, creators, outputs, mean
            case nTokens = "n_tokens"
            case seqWidth = "seq_width"
            case logScale = "log_scale"
            case zLatent = "z_latent"
            case emitMask = "emit_mask"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "encoder", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func array(_ e: Fixture.Entry) -> MLXArray? {
        guard let b64 = e.b64, let d = Data(base64Encoded: b64) else { return nil }
        return MLXArray(d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    // #require does not nest, so each lookup is its own step.
    static func field(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture, "encoder.json missing")
        let e = try #require(fx.fields[k], "no field \(k)")
        return try #require(array(e), "field \(k) was summarised, not stored")
    }

    static func creator(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture, "encoder.json missing")
        let e = try #require(fx.creators[k], "no creator \(k)")
        return try #require(array(e), "creator \(k) was summarised, not stored")
    }

    static func output(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture, "encoder.json missing")
        let e = try #require(fx.outputs[k], "no output \(k)")
        return try #require(array(e), "output \(k) was summarised, not stored")
    }

    /// The structure as upstream's own batch presents it.
    static func structure() throws -> Encoder.Structure {
        // The encoder works in nanometres, and `coords_nm` is not a dataset field:
        // the autoencoder's wrapper divides by 10 before the encoder ever runs.
        let coordsNM = try field("coords") / 10
        let padding = try field("padding_mask")
        let context = try field("is_in_context_conditioning")
        let supports = try field("supports_local_latent")
        let attn = padding * (1 - context)
        return Encoder.Structure(
            coordsNM: coordsNM,
            atom37Mask: try field("atom37_mask"),
            tokenType: try field("token_type"),
            chainIndex: try field("chain_iid"),
            incidentBondTypeCounts: try field("incident_bond_type_counts"),
            // `ground_truth_element`, not `element`: the two differ, −1 against 0 for
            // an unoccupied slot, and only one of them is what the feature reads.
            element: try field("ground_truth_element"),
            charge: try field("charge"),
            attnMask: attn,
            emitMask: attn * supports)
    }

    private func check(_ got: MLXArray, _ want: MLXArray, _ label: String,
                       tol: Float = 1e-4) {
        #expect(got.shape == want.shape, "\(label): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }
        // NaN-aware. An atom37 slot the mask claims but the structure does not
        // model propagates NaN through the whole encoder, upstream included, so the
        // port must put NaN in exactly the same places — and `NaN != NaN` would
        // otherwise report that agreement as total disagreement.
        let gotNaN = MLX.isNaN(got), wantNaN = MLX.isNaN(want)
        #expect(MLX.all(gotNaN .== wantNaN).item(Bool.self),
                "\(label): NaN appears in different places than upstream")
        let finite = MLX.logicalNot(wantNaN)
        let d = MLX.where(finite, MLX.abs(got - want), MLXArray(Float(0)))
        let refMax = max(MLX.where(finite, MLX.abs(want), MLXArray(Float(0)))
                            .max().item(Float.self), 1e-12)
        let rel = d.max().item(Float.self) / refMax
        let detail = String(format: "%@: max |Δ| %.3e  max rel %.3e", label,
                            d.max().item(Float.self), rel)
        #expect(rel <= tol, "\(detail)")
        if rel <= tol { print("ok   encoder \(detail)") }
    }

    @Test("the fixture is a real structure with the encoder's inputs in it")
    func fixtureShape() throws {
        let fx = try #require(Self.fixture, "encoder.json missing")
        #expect(fx.seqWidth == 1587, "concatenated width changed")
        #expect(fx.creators.count == 6)
        // Two creators of the SAME class — absolute and CA-relative coordinates —
        // which is why the fixture keys them by position. Keyed by name, one silently
        // overwrites the other and a feature vanishes from the comparison.
        let coordCreators = fx.creators.values.filter {
            $0.creator == "Atom37NanometersCoorsSeqFeat"
        }
        #expect(coordCreators.count == 2, "the two coordinate creators collapsed into one")
        #expect(MLX.any(try Self.field("atom37_mask") .> 0).item(Bool.self))
    }

    @Test("the encoder's slot widths match the projection the checkpoint carries")
    func slotWidths() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip encoder widths: set RFD4_WEIGHTS"); return
        }
        let w = try ParityWeights.float32(p)
        let e = try Encoder(w)
        #expect(e.seqFactory.inputWidth == 1587,
                "linear_out expects \(e.seqFactory.inputWidth), the features make 1587")
    }

    @Test("each encoder feature matches upstream")
    func perFeature() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip encoder features: set RFD4_WEIGHTS"); return
        }
        let w = try ParityWeights.float32(p)
        let e = try Encoder(w)
        let s = try Self.structure()

        check(e.tokenTypeEmbedding(s.tokenType),
              try Self.creator("0:TokenTypeSeqFeat"), "TokenTypeSeqFeat")
        check(Encoder.atom37Coords(s.coordsNM, atom37Mask: s.atom37Mask, relative: false),
              try Self.creator("1:Atom37NanometersCoorsSeqFeat"), "Atom37Coors (absolute)")
        check(Encoder.atom37Coords(s.coordsNM, atom37Mask: s.atom37Mask, relative: true),
              try Self.creator("2:Atom37NanometersCoorsSeqFeat"), "Atom37Coors (CA-relative)")
        check(s.chainIndex.expandedDimensions(axis: -1).asType(.float32),
              try Self.creator("3:ChainIdxSeqFeat"), "ChainIdxSeqFeat")
        check(Encoder.incidentBondTypeCounts(s.incidentBondTypeCounts,
                                             atom37Mask: s.atom37Mask),
              try Self.creator("4:IncidentBondTypeCountsSeqFeat"), "IncidentBondTypeCounts")
        check(e.elementCharge(s.element, charge: s.charge, atom37Mask: s.atom37Mask),
              try Self.creator("5:Atom37ElementChargeEmbeddingSeqFeat"), "ElementCharge")
    }

    @Test("the whole encoder matches upstream")
    func encoderParity() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip encoder parity: set RFD4_WEIGHTS"); return
        }
        let w = try ParityWeights.float32(p)
        let e = try Encoder(w)
        let s = try Self.structure()
        let fx = try #require(Self.fixture)

        check(e.sequenceFeatures(s), try Self.output("init_repr_factory"),
              "init_repr_factory")

        let out = e(s)
        // The head is stochastic, so parity is against the MEAN and the LOG-SCALE,
        // which are deterministic. Comparing `z` to `z` would compare two different
        // draws and report an error that is not one.
        check(out.mean, try #require(Self.array(fx.mean)), "mean")
        check(out.logScale, try #require(Self.array(fx.logScale)), "log_scale")
        check(out.emitMask, try #require(Self.array(fx.emitMask)), "emit_mask")

        // With no seed the encoder returns the mean itself, so a caller that wants a
        // deterministic encode gets one rather than an undisclosed draw.
        let d = MLX.abs(out.z - out.mean * out.emitMask.expandedDimensions(axis: -1))
        #expect(d.max().item(Float.self) == 0,
                "an unseeded encode drew a sample instead of returning the mean")
    }

    /// An unresolved atom in an occupied slot is refused, not silently encoded.
    ///
    /// This is the one place the port declines to be bug-compatible, and the reason
    /// is that the alternative is not a degraded answer but no answer: a latent that
    /// is entirely NaN, which reaches the sampler, the decoder, and finally a
    /// structure file whose every coordinate is NaN. Measured on upstream's own
    /// shipped binder target, which has six such atoms.
    @Test("a structure with unresolved occupied slots is refused")
    func unresolvedSlotsRefused() throws {
        var s = try Self.structure()
        #expect(throws: Never.self) { try s.validate() }

        // One unresolved atom in a slot the mask claims.
        var coords = s.coordsNM
        coords[0, 3, 5] = MLXArray([Float.nan, .nan, .nan])
        var mask = s.atom37Mask
        mask[0, 3, 5] = MLXArray(Float(1))
        s.coordsNM = coords
        s.atom37Mask = mask
        #expect(throws: EncoderError.self) { try s.validate() }

        // And the reason it is refused: the latent really does go entirely NaN,
        // across every token, not just the one that was unresolved.
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            return
        }
        let w = try ParityWeights.float32(p)
        let out = try Encoder(w)(s)
        #expect(MLX.all(MLX.isNaN(out.mean)).item(Bool.self),
                "one unresolved atom did not poison the whole latent; if that is no longer true the guard can be relaxed")
    }

    /// The draw is a draw: it moves, it is reproducible, and it is scaled by the
    /// predicted log-scale rather than by a constant.
    @Test("the reparameterised sample is not the mean")
    func reparameterisation() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip reparameterisation: set RFD4_WEIGHTS"); return
        }
        let w = try ParityWeights.float32(p)
        let e = try Encoder(w)
        let s = try Self.structure()

        let mean = e(s)
        let a = e(s, seed: 1)
        let b = e(s, seed: 1)
        let c = e(s, seed: 2)

        #expect(MLX.abs(a.z - b.z).max().item(Float.self) == 0,
                "the same seed gave two different samples")
        #expect(MLX.abs(a.z - c.z).max().item(Float.self) > 1e-6,
                "two seeds gave the same sample")
        #expect(MLX.abs(a.z - mean.z).max().item(Float.self) > 1e-6,
                "the sample is the mean — the draw is not happening")

        // The spread must follow exp(log_scale). Compared against the deterministic
        // quantity rather than against a magic number, so this says something about
        // the model rather than about this fixture.
        let emit = mean.emitMask.expandedDimensions(axis: -1)
        let sigma = MLX.exp(mean.logScale) * emit
        let deviation = MLX.abs(a.z - mean.z)
        let live = MLX.sum(emit).item(Float.self)
        #expect(live > 0)
        // |z - mean| = |noise| * sigma, and |noise| averages sqrt(2/pi) ≈ 0.7979.
        let ratio = MLX.sum(deviation).item(Float.self)
            / max(MLX.sum(sigma).item(Float.self), 1e-12)
        #expect(ratio > 0.5 && ratio < 1.2,
                "spread/sigma is \(ratio); the sample is not scaled by exp(log_scale)")
    }
}
