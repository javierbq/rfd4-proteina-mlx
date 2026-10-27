import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// The three factories end to end: `s_init`, `z_init` and `cond` as the trunk
/// receives them.
///
/// This is the test that closes the loop. Each feature being right in isolation
/// says nothing about the concatenation: a factory projects one bias-free matrix
/// across the FULL width including the inert blocks, so a slot of the wrong width,
/// or in the wrong order, misaligns every weight after it while every individual
/// feature still passes.
@Suite("featurizer parity")
struct FeaturizerParityTests {

    typealias Fixture = FeatureParityTests.Fixture

    nonisolated(unsafe) static let fixture = FeatureParityTests.fixture

    static func arr(_ e: Fixture.Entry) -> MLXArray { FeatureParityTests.array(e) }

    static func batch(_ fx: Fixture) throws -> Featurizer.Batch {
        func input(_ k: String) throws -> MLXArray {
            arr(try #require(fx.inputs[k], "fixture has no input \(k)"))
        }
        return Featurizer.Batch(
            xtBBCA: try input("x_t.bb_ca"),
            xtLocalLatents: try input("x_t.local_latents"),
            tBBCA: try input("t.bb_ca").item(Float.self),
            tLocalLatents: try input("t.local_latents").item(Float.self),
            tokenType: try input("token_type"),
            paddingMask: try input("padding_mask"),
            cathCode: try input("cath_code"),
            coordGroup: try input("coord_group"),
            indexGroup: try input("index_group"),
            moleculeIID: try input("molecule_iid"),
            moleculeEntityID: try input("molecule_entity_id"),
            bondOrder: try input("bond_order"),
            tokenBondPathLength: try input("token_bond_path_length")
        )
    }

    private func check(_ got: MLXArray, _ want: MLXArray, _ label: String, tol: Float = 1e-4) {
        #expect(got.shape == want.shape, "\(label): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }
        let d = MLX.abs(got - want)
        let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
        let rel = d.max().item(Float.self) / refMax
        let detail = String(format: "max |Δ| %.3e  max rel %.3e", d.max().item(Float.self), rel)
        #expect(rel <= tol, "\(label): \(detail)")
        if rel <= tol { print("ok   factory \(label): \(detail)") }
    }

    @Test("slot widths match the projections the checkpoint carries")
    func slotWidths() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip slot widths: set RFD4_WEIGHTS"); return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let f = try Featurizer(w)
        // If a declared slot width is wrong the sum will not match linear_out's
        // input dimension, which is the cheapest possible way to catch it.
        #expect(Featurizer.width(Featurizer.seqSlots) == f.seqFactory.inputWidth)
        #expect(Featurizer.width(Featurizer.condSlots) == f.condFactory.inputWidth)
        #expect(Featurizer.width(Featurizer.pairSlots) == f.pairFactory.inputWidth)
        #expect(Featurizer.width(Featurizer.pairCondSlots) == f.pairCondFactory.inputWidth)
    }

    @Test("the three factories match upstream")
    func factoryParity() throws {
        let fx = try #require(Self.fixture, "features.json missing")
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip featurizer parity: set RFD4_WEIGHTS"); return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let f = try Featurizer(w)
        let b = try Self.batch(fx)

        let factories = try #require(fx.factories, "fixture carries no factory outputs")
        check(f.sequenceFeatures(b), Self.arr(try #require(factories["init_repr_factory"])),
              "init_repr_factory")
        // The fixture captures the FACTORY's output. What the trunk receives is
        // that passed through transition_c_1 and c_2, which is checked separately
        // in PredictStepTests against the trunk's own captured input — the two must
        // not be conflated, since that conflation is what hid the missing
        // transitions in the first place.
        check(f.conditioningFactory(b), Self.arr(try #require(factories["cond_factory"])),
              "cond_factory (pre-transition)")
        check(f.pairRepresentation(b), Self.arr(try #require(factories["pair_repr_builder"])),
              "pair_repr_builder")
    }
}
