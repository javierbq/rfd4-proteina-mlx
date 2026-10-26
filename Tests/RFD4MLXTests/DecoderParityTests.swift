import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// Parity for the autoencoder decoder — the half of the model that turns a CA
/// position and a 32-dim latent into all-atom geometry and a residue identity.
///
/// It also cross-checks `RFD4Core`'s vocabulary transcription from the other side:
/// the allowed-residue `[7][147]` and atom `[147][37]` masks are derived here from
/// the YAML, while upstream derives its own from the same source and bakes them into
/// buffers. If the transcription were wrong, the masked logits and `atom_mask` would
/// diverge even with every weight correct.
@Suite("decoder parity")
struct DecoderParityTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable { let shape: [Int]; let b64: String }
        let torch: String
        let nTokens: Int
        let inputs: [String: Entry]
        let outputs: [String: Entry]
        enum CodingKeys: String, CodingKey {
            case torch, inputs, outputs
            case nTokens = "n_tokens"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "decoderio", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func array(_ e: Fixture.Entry) -> MLXArray {
        let data = Data(base64Encoded: e.b64)!
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return MLXArray(floats, e.shape)
    }

    static let tolerance: Float = 2e-3

    private func check(_ got: MLXArray, _ want: MLXArray, _ label: String) {
        #expect(got.shape == want.shape, "\(label): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }
        let d = MLX.abs(got.asType(.float32) - want)
        let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
        let maxAbs = d.max().item(Float.self)
        let rel = maxAbs / refMax
        let detail = String(format: "max |Δ| %.3e  max rel %.3e  (|ref| max %.3e)",
                            maxAbs, rel, refMax)
        #expect(rel <= Self.tolerance, "\(label): \(detail)")
        if rel <= Self.tolerance { print("ok   \(label): \(detail)") }
    }

    @Test("the decoder matches upstream")
    func decoderParity() throws {
        guard let fx = Self.fixture else {
            Issue.record("decoderio.json missing — regenerate with dump_decoder_io.py")
            return
        }
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip decoder parity: set RFD4_WEIGHTS to the checkpoint")
            return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let decoder = try Decoder(w)

        let out = decoder(
            seqInit: Self.array(fx.inputs["seq_init"]!),
            cond: Self.array(fx.inputs["cond"]!),
            attnMask: Self.array(fx.inputs["attn_mask"]!),
            emitMask: Self.array(fx.inputs["emit_mask"]!),
            caCoordsNM: Self.array(fx.inputs["ca_coors_nm"]!),
            tokenType: Self.array(fx.inputs["token_type"]!).asType(.int32)
        )

        check(out.coordsNM, Self.array(fx.outputs["coors_nm"]!), "coors_nm")
        check(out.atomPresenceLogits,
              Self.array(fx.outputs["atom_presence_logits"]!), "atom_presence_logits")

        // seq_logits carries -finfo.min in the forbidden slots, which would swamp a
        // relative comparison. Compare the live classes, then the masking separately.
        let wantLogits = Self.array(fx.outputs["seq_logits"]!)
        let live = wantLogits .> -1e30
        check(MLX.where(live, out.seqLogits, MLXArray(Float(0))),
              MLX.where(live, wantLogits, MLXArray(Float(0))), "seq_logits (allowed)")
        let gotLive = out.seqLogits .> -1e30
        #expect(MLX.all(gotLive .== live).item(Bool.self),
                "seq_logits: the set of allowed residue classes differs from upstream")

        // Discrete outputs must match exactly — these are the decoder's actual answer.
        let wantAatype = Self.array(fx.outputs["aatype_max"]!)
        #expect(MLX.all(out.aatype.asType(.float32) .== wantAatype).item(Bool.self),
                "aatype_max differs from upstream")
        let wantAtomMask = Self.array(fx.outputs["atom_mask"]!) .> 0
        #expect(MLX.all(out.atomMask .== wantAtomMask).item(Bool.self),
                "atom_mask differs from upstream")
    }

    @Test("the decoder has twelve layers and no pair track")
    func decoderShape() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let decoder = try Decoder(w)
        #expect(decoder.layers.count == 12)
        // use_attn_pair_bias is false, so no layer has a pair-bias projection.
        for (i, layer) in decoder.layers.enumerated() {
            #expect(layer.mhba.mha.biasW == nil, "decoder layer \(i) should have no pair bias")
        }
    }
}
