import Foundation
import MLX
import MLXRandom
import RFD4Core
import Testing
@testable import RFD4MLX

/// The gap `BinderFeaturizerTests` leaves open.
///
/// That suite builds the batch from the FIXTURE's arrays, so it proves the
/// featurizer is right *given correct inputs*. It cannot prove the port produces
/// correct inputs — and the port producing wrong inputs is exactly the failure mode
/// that has cost this work the most: a preserved segment the wrong length, a
/// coordinate group the wrong shape, a channel list in the wrong order.
///
/// This closes it. Everything here comes from `DesignProblem` — the layout, the
/// grouping ids, the connectivity, the mask derivation, the channel order, the
/// coordinate-group densification, the concatenation — and the three factory outputs
/// are compared against upstream's.
///
/// One thing is necessarily substituted: upstream's internal coordinates. Their
/// frame is a **free rotation**, so the port and upstream are both right and
/// numerically different; comparing them would be testing the random number
/// generator. The rotation is checked separately for what it must actually preserve
/// (shape and centring) in `BinderFeaturizerTests.designProblemAssembly`.
@Suite("binder assembly")
struct BinderAssemblyTests {

    static func field(_ k: String) throws -> MLXArray { try BinderLayoutTests.field(k) }

    /// Rebuild the design problem from the structural facts, not from the batch.
    @Test("the port's own assembly reproduces upstream's trunk inputs")
    func assemblyDrivesFeaturizer() throws {
        let fx = try #require(BinderLayoutTests.fixture, "binder.json missing")
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip binder assembly: set RFD4_WEIGHTS"); return
        }
        let layout = BinderLayoutTests.layout()
        let slots = Vocabulary.slotCount
        let n = layout.count

        // What the structure says, read only from arrays a parser would produce:
        // which atom37 slots hold an atom, where they are, which carry a coordinate
        // group, which are hotspots, and what each residue is.
        let atom37 = try Self.field("atom37_mask")[0].asArray(Float.self)
        let groups = try Self.field("condition_CoordinateGroup_1_atom")[0].asArray(Float.self)
        let hotspotFlags = try Self.field("condition_hotspot_1_atom")[0].asArray(Float.self)
        let residueType = try Self.field("residue_type")[0].asArray(Float.self)
        let upstreamInternal = try Self.field("condition_InternalCoordinate_1_atom")

        // Coordinates: any finite placeholder will do for the slots that have atoms,
        // since the internal coordinates are substituted below and nothing else in
        // the featurizer reads raw positions. Using the real ones would make this
        // test depend on a CIF it does not need.
        var coords = [[Vec3?]](repeating: [Vec3?](repeating: nil, count: slots), count: n)
        var members: [(token: Int, slot: Int)] = []
        var hotspots: [(token: Int, slot: Int)] = []
        for t in 0 ..< n {
            for s in 0 ..< slots {
                let i = t * slots + s
                guard atom37[i] > 0 else { continue }
                coords[t][s] = Vec3(Float(t) * 3.8, Float(s) * 1.4, Float((t &* 7 &+ s) % 11))
                if groups[i] > 0 { members.append((t, s)) }
                if hotspotFlags[i] > 0 { hotspots.append((t, s)) }
            }
        }
        #expect(members.count == 460, "coordinate group is not the target's backbone")
        #expect(hotspots.count == 6, "expected the recipe's six hotspot atoms")

        // The target's sequence, mapped through the port's OWN channel list. A token
        // counts as known when the structure gives it atoms — which is the target —
        // so nothing here is read back from the sequence-probability arrays this is
        // meant to reproduce.
        //
        // `residue_type` is upstream's canonical-20 alphabet, alphabetical by
        // three-letter code — and it is **1-based**: 1 = ALA through 20 = VAL, with 0
        // for a token whose residue is not given. Measured against the fixture's own
        // sequence-probability channels, every one of the twenty. Reading it as
        // 0-based would shift every sequence condition by one residue type, which the
        // factory comparison below does catch, but only after the fact.
        var knownSequence = [Int?](repeating: nil, count: n)
        for t in 0 ..< n {
            let hasAtoms = (0 ..< slots).contains { atom37[t * slots + $0] > 0 }
            guard hasAtoms else { continue }
            let cls = Int(residueType[t])
            guard cls >= 1, cls <= 20 else { continue }
            knownSequence[t] = cls - 1
        }
        #expect(knownSequence.compactMap { $0 }.count == 115,
                "expected the target's 115 residues to carry a known sequence")

        let problem = DesignProblem(
            layout: layout, coords: coords,
            internalGroups: [DesignProblem.CoordinateGroup(members: members)],
            hotspots: hotspots, knownSequence: knownSequence)
        // `system`, not `validatedSystem`: the placeholder coordinates above are
        // spread to be non-degenerate rather than to be chemically plausible, so the
        // peptide-bond check would reject them. That check is exercised against real
        // geometry in `BondPathLengthTests`.
        var system = try problem.system(seed: 0)

        // Substitute upstream's frame for the port's; see the suite comment.
        var conditioning = try #require(system.conditioning)
        conditioning.internalCoords = upstreamInternal
        system.conditioning = conditioning

        let w = try ParityWeights.float32(p)
        let f = try Featurizer(w)
        let b = Featurizer.Batch(
            xtBBCA: try Self.field("x_t.bb_ca"),
            xtLocalLatents: try Self.field("x_t.local_latents"),
            tBBCA: try Self.field("t.bb_ca").item(Float.self),
            tLocalLatents: try Self.field("t.local_latents").item(Float.self),
            tokenType: system.tokenType, paddingMask: system.paddingMask,
            cathCode: system.cathCode, coordGroup: system.coordGroup,
            indexGroup: system.indexGroup, moleculeIID: system.moleculeIID,
            moleculeEntityID: system.moleculeEntityID, bondOrder: system.bondOrder,
            tokenBondPathLength: system.tokenBondPathLength,
            canonicalPolymerOrdering: system.canonicalPolymerOrdering,
            conditioning: conditioning)

        func check(_ got: MLXArray, _ want: MLXArray, _ label: String) {
            #expect(got.shape == want.shape, "\(label): \(got.shape) != \(want.shape)")
            guard got.shape == want.shape else { return }
            let d = MLX.abs(got - want)
            let refMax = max(MLX.abs(want).max().item(Float.self), 1e-12)
            let rel = d.max().item(Float.self) / refMax
            let detail = String(format: "%@: max rel %.3e", label, rel)
            #expect(rel <= 1e-4, "\(detail)")
            if rel <= 1e-4 {
                print(String(format: "ok   assembly %@: max rel %.3e", label, rel))
            }
        }

        let seq = try #require(fx.factories["init_repr_factory"])
        check(f.sequenceFeatures(b), try #require(BinderLayoutTests.array(seq)),
              "init_repr_factory")
        let cond = try #require(fx.factories["cond_factory"])
        check(f.conditioningFactory(b), try #require(BinderLayoutTests.array(cond)),
              "cond_factory")
        let pair = try #require(fx.factories["pair_repr_builder"])
        let corner = try #require(pair.corner)
        check(f.pairRepresentation(b)[0..., 0 ..< corner, 0 ..< corner, 0...],
              try #require(BinderLayoutTests.array(pair)),
              "pair_repr_builder (\(corner)×\(corner) corner)")
    }

    /// The rotation must be a rotation — orthonormal with determinant `+1`.
    ///
    /// A reflection preserves every pairwise distance, so the shape check in
    /// `designProblemAssembly` would pass for one. It would also mirror the target,
    /// turning an L-protein into its enantiomer: the one error in this code that a
    /// distance comparison provably cannot see.
    @Test("the random rotation is a rotation, not a reflection")
    func rotationIsProper() throws {
        for seed in UInt64(0) ..< 32 {
            let r = DesignProblem.randomRotation(key: MLXRandom.key(seed))
            #expect(r.count == 9)

            func dot(_ a: Int, _ b: Int) -> Float {
                (0 ..< 3).reduce(Float(0)) { $0 + r[a * 3 + $1] * r[b * 3 + $1] }
            }
            for i in 0 ..< 3 {
                #expect(abs(dot(i, i) - 1) < 1e-4, "row \(i) is not a unit vector (seed \(seed))")
                for j in (i + 1) ..< 3 {
                    #expect(abs(dot(i, j)) < 1e-4, "rows \(i),\(j) not orthogonal (seed \(seed))")
                }
            }
            let det = r[0] * (r[4] * r[8] - r[5] * r[7])
                - r[1] * (r[3] * r[8] - r[5] * r[6])
                + r[2] * (r[3] * r[7] - r[4] * r[6])
            #expect(abs(det - 1) < 1e-4,
                    "determinant \(det) at seed \(seed) — a reflection would mirror the target")
        }
    }

    /// Different seeds must give different frames, or the "random" rotation is a
    /// constant and the pose is not free at all.
    @Test("the rotation actually varies with the seed")
    func rotationVaries() throws {
        let a = DesignProblem.randomRotation(key: MLXRandom.key(1))
        let b = DesignProblem.randomRotation(key: MLXRandom.key(2))
        let spread = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        #expect(spread > 1e-3, "two seeds gave the same rotation")
        // …and the same seed must give the same frame, or a design is unreproducible.
        let again = DesignProblem.randomRotation(key: MLXRandom.key(1))
        #expect(zip(a, again).allSatisfy { $0 == $1 }, "the same seed gave two frames")
    }
}
