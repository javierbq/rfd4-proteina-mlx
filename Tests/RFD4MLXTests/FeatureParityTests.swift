import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// Per-feature parity: each creator checked alone, against its own captured output.
///
/// The factories hold 35 creators and only 13 carry signal for an unconditioned
/// canonical-20 protein. Checking the factory output as a whole would let a wrong
/// feature hide behind a projection; checking each one separately does not.
@Suite("feature parity")
struct FeatureParityTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable {
            let shape: [Int]; let b64: String; let dead: Bool
        }
        let torch: String
        let inputs: [String: Entry]
        let features: [String: Entry]
        let factories: [String: Entry]?
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "features", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func array(_ e: Fixture.Entry) -> MLXArray {
        let data = Data(base64Encoded: e.b64)!
        return MLXArray(data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    static func input(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture); return array(try #require(fx.inputs[k]))
    }
    static func feature(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture); return array(try #require(fx.features[k], "no feature \(k)"))
    }

    private func check(_ got: MLXArray, _ want: MLXArray, _ label: String, tol: Float = 1e-5) {
        #expect(got.shape == want.shape, "\(label): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }
        let d = MLX.abs(got - want)
        let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
        let rel = d.max().item(Float.self) / refMax
        let detail = String(format: "max |Δ| %.3e  max rel %.3e", d.max().item(Float.self), rel)
        #expect(rel <= tol, "\(label): \(detail)")
        if rel <= tol { print("ok   feat \(label): \(detail)") }
    }

    // MARK: live features with no learned weights

    @Test("x_t passthrough features")
    func xtPassthrough() throws {
        check(try Self.input("x_t.bb_ca"), try Self.feature("init_repr_factory/XtBBCASeqFeat"),
              "XtBBCASeqFeat")
        check(try Self.input("x_t.local_latents"),
              try Self.feature("init_repr_factory/XtLocalLatentsSeqFeat"), "XtLocalLatentsSeqFeat")
    }

    /// Measured 2.97e-5; the budget is the next round number above it.
    ///
    /// NOT slop. `t` is scaled by 2000 before the frequencies apply, so the
    /// lowest-frequency angle reaches ~1000, where one float32 ULP is 1.2e-4. The
    /// residual is under HALF an ULP of that angle — a sub-ULP disagreement between
    /// libm's `expf`/`sinf` and torch's vectorized ones, showing through at full
    /// scale because sin is being evaluated 159 periods from the origin.
    ///
    /// Replaying torch's exact op sequence reproduces its output bit for bit, and
    /// computing the angles in double makes it *worse* (4.4e-5), which confirms the
    /// arithmetic is right and only the transcendentals differ. 3e-5 on values in
    /// [-1, 1] is an order of magnitude under the int8 floor this port targets.
    static let timeEmbeddingTolerance: Float = 1e-4

    @Test("sinusoidal time embedding")
    func timeEmbedding() throws {
        // The two creators differ only in which modality's t they read.
        let tBB = try Self.input("t.bb_ca").item(Float.self)
        let tLL = try Self.input("t.local_latents").item(Float.self)
        check(Features.timeEmbedding(tBB, dim: 256),
              try Self.feature("cond_factory/TimeEmbeddingBatchFeat"), "TimeEmbedding(bb_ca)",
              tol: Self.timeEmbeddingTolerance)
        check(Features.timeEmbedding(tLL, dim: 256),
              try Self.feature("cond_factory/TimeEmbeddingBatchFeat#2"),
              "TimeEmbedding(local_latents)", tol: Self.timeEmbeddingTolerance)
    }

    @Test("binned pairwise CA distances")
    func pairwiseDistances() throws {
        // 30 bins; the released limits come from the nn config.
        let x = try Self.input("x_t.bb_ca")
        let want = try Self.feature("pair_repr_builder/XtBBCAPairwiseDistancesPairFeat")
        // Try the documented default range; if the config differs the one-hot will
        // land in the wrong column and this fails loudly rather than approximately.
        let got = Features.binPairwiseDistances(x, minDist: 0.1, maxDist: 3.0, dim: 30)
        check(got, want, "XtBBCAPairwiseDistancesPairFeat")
    }

    @Test("sameness pair features are all the same operation")
    func samenessPairs() throws {
        // All three are `a[:,:,None] == a[:,None,:]`. For a single-chain de-novo run
        // every token shares a molecule and entity, so these are all-ones — which is
        // worth asserting, because an all-ones result also hides a broken comparison.
        for key in ["IndexGroupPairFeat", "MoleculeIdxPairFeat", "MoleculeEntityPairFeat"] {
            let want = try Self.feature("pair_repr_builder/\(key)")
            #expect(want.shape == [1, 8, 8, 1])
            let uniform = MLX.all(want .== want[0, 0, 0, 0]).item(Bool.self)
            #expect(uniform, "\(key): expected a uniform block for a single-chain run")
            // The operation itself, on a grouping that is NOT uniform.
            let groups = MLXArray([Int32(0), 0, 1, 1, 2, 2, 2, 3], [1, 8])
            let same = Features.samenessPair(groups)
            #expect(same[0, 0, 1, 0].item(Float.self) == 1.0)
            #expect(same[0, 0, 2, 0].item(Float.self) == 0.0)
            #expect(same[0, 4, 6, 0].item(Float.self) == 1.0)
        }
    }

    // MARK: the inert 22

    @Test("the features that should be inert really are")
    func inertFeatures() throws {
        let fx = try #require(Self.fixture)
        let expectedInert = [
            "init_repr_factory/SequenceProbabilitiesSeqFeat",
            "init_repr_factory/ConditionalAtom37NanometersCoorsSeqFeat",
            "init_repr_factory/RasaBinnedSeqFeat",
            "init_repr_factory/SecondaryStructureSeqFeat",
            "init_repr_factory/HotspotSeqFeat",
            "init_repr_factory/ChiralitySeqFeat",
            "cond_factory/RecycleRemainingBatchFeat",
            "pair_repr_builder/BondMaskPairFeat",
            "pair_repr_builder/CanonicalPolymerOrderingPairFeat",
        ]
        for key in expectedInert {
            let e = try #require(fx.features[key], "no capture for \(key)")
            #expect(e.dead, "\(key) was expected to be inert for de-novo canonical-20")
        }
        // RecycleRemaining being inert confirms it one-hots to a constant under
        // recycle.grad=last, so the trunk can hard-code it.
        #expect(fx.features["cond_factory/RecycleRemainingBatchFeat"]?.dead == true)

        let live = fx.features.values.filter { !$0.dead }.count
        #expect(live == 13, "expected 13 live creators, found \(live)")
        #expect(fx.features.count == 35)
    }

    @Test("one-hot and binning helpers")
    func helpers() {
        let oh = Features.oneHot(MLXArray([Int32(0), 2, 1]), classes: 3)
        #expect(oh.shape == [3, 3])
        #expect(oh[0, 0].item(Float.self) == 1.0)
        #expect(oh[1, 2].item(Float.self) == 1.0)
        #expect(oh[2, 1].item(Float.self) == 1.0)

        // d-1 limits give d bins: below the first, between, above the last.
        let b = Features.binAndOneHot(MLXArray([Float(-1), 0.5, 5.0]), limits: [0.0, 1.0])
        #expect(b.shape == [3, 3])
        #expect(b[0, 0].item(Float.self) == 1.0, "below the first limit")
        #expect(b[1, 1].item(Float.self) == 1.0, "between limits")
        #expect(b[2, 2].item(Float.self) == 1.0, "above the last limit")
    }
}
