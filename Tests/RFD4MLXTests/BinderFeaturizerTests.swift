import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// The whole featurizer on a real binder design, against upstream's own output.
///
/// `BinderLayoutTests` checks that the port builds the same layout; this checks what
/// the model is handed because of it. The three factory outputs are the only place
/// a conditioning mistake has to show: each factory projects one bias-free matrix
/// across its full concatenation, inert blocks included, so a slot of the wrong
/// width — or a conditioning feature left as zeros — misaligns every weight after it
/// while each individual feature still passes its own test.
///
/// The de-novo suite cannot reach any of this. All four conditioning features are
/// identically zero without a target, so an unimplemented one passes every
/// unconditioned test there is.
@Suite("binder featurizer")
struct BinderFeaturizerTests {

    typealias Fixture = BinderLayoutTests.Fixture

    nonisolated(unsafe) static let fixture = BinderLayoutTests.fixture

    static func field(_ k: String) throws -> MLXArray { try BinderLayoutTests.field(k) }

    /// Drive the Swift featurizer from upstream's own arrays.
    ///
    /// The conditioning payload is taken from the fixture rather than rebuilt from
    /// the CIF, so this isolates the featurizer: if it disagrees, the fault is in the
    /// features or their concatenation, not in how a target was read off disk.
    static func batch(_ fx: Fixture) throws -> Featurizer.Batch {
        let rawGroups = try field("condition_CoordinateGroup_1_atom")
        let densified = CoordGroups.densified(
            rawGroups.reshaped(-1).asArray(Float.self).map { Int($0) })

        var channels: [MLXArray] = []
        for name in fx.seqProbKeys {
            channels.append(try field("condition_sequence-probability-\(name)_1_residue"))
        }

        let conditioning = Featurizer.Batch.Conditioning(
            internalCoords: try field("condition_InternalCoordinate_1_atom"),
            absoluteCoords: try field("condition_coordinate_1_atom"),
            coordGroups: rawGroups,
            coordGroupsDensified: MLXArray(densified.map(Float.init), rawGroups.shape),
            sequenceProbabilities: channels,
            hotspot: try field("condition_hotspot_1_atom"),
            atom37Mask: try field("atom37_mask"))

        return Featurizer.Batch(
            xtBBCA: try field("x_t.bb_ca"),
            xtLocalLatents: try field("x_t.local_latents"),
            tBBCA: try field("t.bb_ca").item(Float.self),
            tLocalLatents: try field("t.local_latents").item(Float.self),
            tokenType: try field("token_type"),
            paddingMask: try field("padding_mask"),
            cathCode: try field("condition_cath-label_1_residue"),
            coordGroup: rawGroups,
            indexGroup: try field("condition_index-group_1_token"),
            moleculeIID: try field("molecule_iid"),
            moleculeEntityID: try field("molecule_entity_id"),
            bondOrder: try field("bond_order"),
            tokenBondPathLength: try field("token_bond_path_length"),
            canonicalPolymerOrdering: try field("canonical_polymer_ordering"),
            conditioning: conditioning)
    }

    private func check(_ got: MLXArray, _ want: MLXArray, _ label: String, tol: Float = 1e-4) {
        #expect(got.shape == want.shape, "\(label): shape \(got.shape) != \(want.shape)")
        guard got.shape == want.shape else { return }
        let d = MLX.abs(got - want)
        let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
        let rel = d.max().item(Float.self) / refMax
        let detail = String(format: "max |Δ| %.3e  max rel %.3e", d.max().item(Float.self), rel)
        #expect(rel <= tol, "\(label): \(detail)")
        if rel <= tol { print("ok   binder \(label): \(detail)") }
    }

    @Test("the conditioning really is live in this fixture")
    func conditioningIsLive() throws {
        // Otherwise every check below compares zeros to zeros and passes while
        // testing nothing — which is exactly how the unconditioned fixtures let an
        // unimplemented conditioning feature through.
        let internalCoords = try Self.field("condition_InternalCoordinate_1_atom")
        #expect(MLX.any(MLX.isFinite(internalCoords)).item(Bool.self),
                "no finite internal coordinates — the target is not in the fixture")
        #expect(MLX.any(try Self.field("condition_hotspot_1_atom") .> 0).item(Bool.self))
        let groups = try Self.field("condition_CoordinateGroup_1_atom")
        #expect(MLX.any(groups .> 0).item(Bool.self), "no internal coordinate group")
        // The absolute feature is the other mode and must be inert here, or the two
        // are being conflated.
        let absolute = try Self.field("condition_coordinate_1_atom")
        #expect(!MLX.any(MLX.isFinite(absolute)).item(Bool.self),
                "this recipe leaves the pose free; absolute coordinates must be absent")
    }

    @Test("coordinate groups are densified, not passed through raw")
    func coordGroupDensification() throws {
        let raw = try Self.field("condition_CoordinateGroup_1_atom")
            .reshaped(-1).asArray(Float.self).map { Int($0) }
        let densified = CoordGroups.densified(raw)
        // One internal group, so every conditioned slot must read 0 — the value
        // upstream's randperm can only produce for k = 1. Passing the raw id would
        // feed 1, which no training example carried.
        #expect(Set(raw) == [-1, 1], "fixture has more than one coordinate group")
        #expect(Set(densified) == [-1, 0], "densification did not relabel group 1 to 0")
        #expect(zip(raw, densified).allSatisfy { ($0 == -1) == ($1 == -1) },
                "densification moved an unconditioned slot")
    }

    /// Each conditioning feature on its own, before they are concatenated.
    ///
    /// When the sequence factory disagrees there are nine live creators it could be,
    /// and the factory output says nothing about which. These are the four that only
    /// a conditioned run can exercise.
    @Test("each conditioning feature matches upstream")
    func perFeature() throws {
        let fx = try #require(Self.fixture, "binder.json missing")
        let b = try Self.batch(fx)
        let c = try #require(b.conditioning)

        // #require does not nest, so each lookup is its own step.
        func upstream(_ name: String) throws -> MLXArray {
            let rec = try #require(fx.creators["init_repr_factory/\(name)"], "no creator \(name)")
            let out = try #require(rec.output, "\(name) has no captured output")
            return try #require(BinderLayoutTests.array(out), "\(name) output was summarised")
        }
        func internalCoords() throws -> MLXArray {
            try #require(c.internalCoords, "no internal coordinates")
        }
        func absoluteCoords() throws -> MLXArray {
            try #require(c.absoluteCoords, "no absolute coordinates")
        }

        check(Features.sequenceProbabilities(c.sequenceProbabilities),
              try upstream("SequenceProbabilitiesSeqFeat"), "SequenceProbabilitiesSeqFeat")
        check(Features.conditionalAtom37InternalCoords(
                try internalCoords(), coordGroups: c.coordGroups),
              try upstream("ConditionalAtom37NanometersInternalCoorsSeqFeat"),
              "ConditionalAtom37NanometersInternalCoorsSeqFeat")
        check(Features.hotspot(c.hotspot, atom37Mask: c.atom37Mask),
              try upstream("HotspotSeqFeat"), "HotspotSeqFeat")
        check(c.coordGroupsDensified, try upstream("CoordGroupSeqFeat"), "CoordGroupSeqFeat")
        // This recipe leaves the pose free, so BOTH sides of this comparison are
        // identically zero and it cannot catch an error inside the feature — the
        // scaling, the mask, the flattening. It is kept for what it does establish:
        // that an all-NaN input produces an all-zero block of the right shape rather
        // than NaNs, which would poison the whole concatenation. The feature's
        // arithmetic is covered where that creator is live, in
        // `ConditioningParityTests.conditionalCoords`.
        let absoluteBlock = Features.conditionalAtom37Coords(try absoluteCoords())
        check(absoluteBlock, try upstream("ConditionalAtom37NanometersCoorsSeqFeat"),
              "ConditionalAtom37NanometersCoorsSeqFeat (inert here — shape only)")
        #expect(MLX.all(MLX.isFinite(absoluteBlock)).item(Bool.self),
                "an all-NaN input produced NaNs rather than an inert zero block")
    }

    /// The emit masks are a formula, not the all-ones they currently evaluate to.
    @Test("emit masks follow compute_flow_emit_masks, not a constant")
    func emitMaskFormula() throws {
        let layout = BinderLayoutTests.layout()
        let n = layout.count
        let coords = [[Vec3?]](repeating: [Vec3?](repeating: nil, count: Vocabulary.slotCount),
                               count: n)
        let known = [Int?](repeating: nil, count: n)

        // The binder case: nothing is context, everything supports a local latent.
        let plain = try DesignProblem(layout: layout, coords: coords, knownSequence: known)
            .system(seed: 0)
        #expect(MLX.all(plain.emitBBCA .== 1).item(Bool.self))
        #expect(MLX.all(plain.emitLocalLatents .== 1).item(Bool.self))

        // Partial diffusion holds the first chain as context, and one token is
        // atomized. If the masks were the constant they evaluate to above, neither
        // of these would show.
        var context = [Bool](repeating: false, count: n)
        for t in 0 ..< 110 { context[t] = true }
        var supports = [Bool](repeating: true, count: n)
        supports[200] = false

        let partial = try DesignProblem(
            layout: layout, coords: coords, knownSequence: known,
            isContext: context, supportsLocalLatent: supports).system(seed: 0)
        let bb = partial.emitBBCA[0].asArray(Float.self)
        let ll = partial.emitLocalLatents[0].asArray(Float.self)
        #expect(bb[0] == 0 && bb[109] == 0, "context tokens still emit bb_ca")
        #expect(bb[110] == 1 && bb[224] == 1, "non-context tokens stopped emitting")
        // local_latents is the bb_ca mask AND supports_local_latent, so the atomized
        // token drops out of it while still emitting bb_ca.
        #expect(bb[200] == 1 && ll[200] == 0,
                "supports_local_latent is not being applied to the latent mask")
        #expect(ll[110] == 1)
    }

    @Test("the three factories match upstream on a binder")
    func factoryParity() throws {
        let fx = try #require(Self.fixture, "binder.json missing")
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip binder featurizer parity: set RFD4_WEIGHTS"); return
        }
        let w = try Weights.load(URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let f = try Featurizer(w)
        let b = try Self.batch(fx)

        let seq = try #require(fx.factories["init_repr_factory"])
        check(f.sequenceFeatures(b), try #require(BinderLayoutTests.array(seq)),
              "init_repr_factory")

        let cond = try #require(fx.factories["cond_factory"])
        check(f.conditioningFactory(b), try #require(BinderLayoutTests.array(cond)),
              "cond_factory (pre-transition)")

        // The pair representation is [1, 225, 225, 384] — 78 MB — so the fixture
        // stores a corner. A misaligned concatenation is visible in the first token
        // pair as clearly as in the last, and the de-novo fixture checks the whole
        // array at 8 tokens.
        let pair = try #require(fx.factories["pair_repr_builder"])
        let corner = try #require(pair.corner)
        let got = f.pairRepresentation(b)[0..., 0 ..< corner, 0 ..< corner, 0...]
        check(got, try #require(BinderLayoutTests.array(pair)),
              "pair_repr_builder (\(corner)×\(corner) corner)")
    }

    @Test("a design problem assembles into the same conditioning")
    func designProblemAssembly() throws {
        // The layout half of the pipeline, end to end: build a DesignProblem the way
        // a caller would and check the arrays it produces agree with upstream's
        // wherever the problem statement determines them. The internal coordinates
        // are deliberately excluded — their frame is a free rotation, so the port and
        // upstream will not agree on the numbers and must not be made to.
        let layout = BinderLayoutTests.layout()
        let slots = Vocabulary.slotCount
        let atom37 = try Self.field("atom37_mask")[0].asArray(Float.self)
        let hotspot = try Self.field("condition_hotspot_1_atom")[0].asArray(Float.self)
        let groups = try Self.field("condition_CoordinateGroup_1_atom")[0].asArray(Float.self)

        var coords = [[Vec3?]](repeating: [Vec3?](repeating: nil, count: slots),
                               count: layout.count)
        var members: [(token: Int, slot: Int)] = []
        var hots: [(token: Int, slot: Int)] = []
        for t in 0 ..< layout.count {
            for s in 0 ..< slots {
                let i = t * slots + s
                // NOT coplanar. With z fixed at 0 every point lies in one plane, and
                // a reflection through that plane preserves all of them — so the
                // distance check below would pass for a mirrored target, which is the
                // one error it exists to catch. `BinderAssemblyTests.rotationIsProper`
                // checks the determinant directly; this makes the geometry able to
                // see it too.
                if atom37[i] > 0 {
                    coords[t][s] = Vec3(Float(t) * 1.7, Float(s) * 0.9,
                                        Float((t &* 13 &+ s &* 7) % 23) * 0.6)
                }
                if groups[i] > 0 { members.append((t, s)) }
                if hotspot[i] > 0 { hots.append((t, s)) }
            }
        }
        var knownSequence = [Int?](repeating: nil, count: layout.count)
        for t in 110 ..< layout.count { knownSequence[t] = 1 }   // the target: ALA

        let problem = DesignProblem(
            layout: layout, coords: coords,
            internalGroups: [DesignProblem.CoordinateGroup(members: members)],
            hotspots: hots, knownSequence: knownSequence)
        let system = try problem.system(seed: 0)
        let c = try #require(system.conditioning)

        #expect(system.tokenCount == 225)
        let pathDiff = MLX.abs(system.tokenBondPathLength
                               - (try Self.field("token_bond_path_length"))).max().item(Float.self)
        #expect(pathDiff == 0, "path lengths differ by \(pathDiff)")
        let hotDiff = MLX.abs(c.hotspot - (try Self.field("condition_hotspot_1_atom")))
            .max().item(Float.self)
        #expect(hotDiff == 0, "hotspots differ by \(hotDiff)")
        let maskDiff = MLX.abs(c.atom37Mask - (try Self.field("atom37_mask")))
            .max().item(Float.self)
        #expect(maskDiff == 0, "atom37 mask differs by \(maskDiff)")
        // Raw group ids follow the C_CRDGRP contract; the densified ones are what
        // the sequence track reads.
        #expect(MLX.all(c.coordGroups .== (try Self.field("condition_CoordinateGroup_1_atom")))
            .item(Bool.self), "coordinate groups differ")

        // The internal coordinates are centred and rotated, so what must hold is that
        // the shape survived: every pairwise distance is preserved and the centroid
        // is the origin. That is the whole content of "internal".
        let ic = try #require(c.internalCoords)
        let picked = members.prefix(64)
        var centroid = (Float(0), Float(0), Float(0))
        var rotated: [(Float, Float, Float)] = []
        for m in picked {
            let v = ic[0, m.token, m.slot].asArray(Float.self)
            rotated.append((v[0], v[1], v[2]))
        }
        for m in members {
            let v = ic[0, m.token, m.slot].asArray(Float.self)
            centroid = (centroid.0 + v[0], centroid.1 + v[1], centroid.2 + v[2])
        }
        let k = Float(members.count)
        #expect(abs(centroid.0 / k) < 1e-3 && abs(centroid.1 / k) < 1e-3
                && abs(centroid.2 / k) < 1e-3, "internal coordinates are not centred")
        for (a, ma) in zip(rotated, picked) {
            for (b, mb) in zip(rotated, picked) {
                let pa = coords[ma.token][ma.slot]!, pb = coords[mb.token][mb.slot]!
                let want = ((pa.x - pb.x) * (pa.x - pb.x) + (pa.y - pb.y) * (pa.y - pb.y)
                            + (pa.z - pb.z) * (pa.z - pb.z)).squareRoot()
                let got = ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1)
                           + (a.2 - b.2) * (a.2 - b.2)).squareRoot()
                #expect(abs(got - want) < 2e-3,
                        "rotation changed a distance: \(got) vs \(want)")
            }
        }
    }
}
