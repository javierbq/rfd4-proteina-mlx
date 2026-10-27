import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// Compares this port's trajectory against a reference trace recorded from
/// upstream, at MATCHING recycle depth.
///
/// The earlier comparison was unfair: upstream's probe ran depth 1 at every step
/// while the Swift run used `thinkingLow`'s 120-pass schedule. Different depth is
/// a different vector field, so that difference had to be eliminated before any
/// conclusion about the dynamics could be drawn.
@Suite("trajectory comparison")
struct TrajectoryComparisonTests {

    /// Rg about the per-axis mean, and consecutive CA distances — in nanometres.
    static func geometry(_ ca: MLXArray) -> (rg: Float, caca: Float) {
        let c = ca.ndim == 3 ? ca[0] : ca
        let n = c.dim(0)
        let centred = c - c.mean(axis: 0, keepDims: true)
        let rg = MLX.sqrt(MLX.mean(MLX.sum(centred * centred, axis: -1))).item(Float.self)
        let d = c[1...] - c[..<(n - 1)]
        let caca = MLX.sqrt(MLX.sum(d * d, axis: -1)).mean().item(Float.self)
        return (rg, caca)
    }

    /// Same seed must give the same sample.
    ///
    /// This failed twice before it passed, both times for reasons invisible from
    /// the generation code: `MLXRandom.seed` sets process-global state that other
    /// work interleaves with, and `Dictionary` iteration order is not stable across
    /// processes, so the two modalities drew in a different order each run.
    ///
    /// It matters beyond tidiness. The playbook's accuracy work compares two runs
    /// that must differ ONLY in precision; with an unreproducible sampler those are
    /// two different draws and any difference is unattributable.
    @Test("the same seed gives the same sample")
    func reproducible() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip reproducibility: set RFD4_WEIGHTS"); return
        }
        let gen = try Generator(weightsAt: URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let system = Generator.System.deNovo(length: 8)
        let depths = Array(repeating: 1, count: 12)

        let a = try gen.generate(system, seed: 7, recycleDepths: depths)
        let b = try gen.generate(system, seed: 7, recycleDepths: depths)
        let c = try gen.generate(system, seed: 8, recycleDepths: depths)

        let same = MLX.abs(a.xFinal["bb_ca"]! - b.xFinal["bb_ca"]!).max().item(Float.self)
        #expect(same == 0, "same seed differed by \(same)")
        // And a different seed must actually give a different sample, or the
        // first assertion would pass for the wrong reason.
        let diff = MLX.abs(a.xFinal["bb_ca"]! - c.xFinal["bb_ca"]!).max().item(Float.self)
        #expect(diff > 1e-4, "different seeds produced the same sample")
    }

    @Test("at matching depth, the trajectory should behave like upstream's")
    func matchedDepth() throws {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty else {
            print("skip trajectory comparison: set RFD4_WEIGHTS"); return
        }
        let gen = try Generator(weightsAt: URL(fileURLWithPath: (p as NSString).expandingTildeInPath))
        let n = 16
        let nsteps = ReleasedRecycleStrategy.nsteps

        // Exactly what the upstream probe used.
        let r = try gen.generate(Generator.System.deNovo(length: n),
                                 seed: 1,
                                 recycleDepths: Array(repeating: 1, count: nsteps))
        let g = Self.geometry(r.xFinal["bb_ca"]!)

        // Upstream, same batch, same depth, 60 steps (trace_upstream_run.py):
        //   start   Rg 1.583 nm  CA-CA 2.091 nm
        //   final   Rg 0.718 nm  CA-CA 0.375 nm (min 0.371 max 0.377)
        //
        // These replace an earlier pair (0.856 / 1.064) recorded when the probe's
        // batch left every connectivity field at the schema default. That batch was
        // not a chain, so neither implementation was being asked for one and the
        // reference encoded the defect. 0.375 nm is the real CA-CA spacing of a
        // protein backbone; 1.064 nm was a cloud of unbonded fragments.
        print(String(format: "     matched depth 1: final Rg %.3f nm  CA-CA %.3f nm "
                     + "(upstream 0.718 / 0.375)", g.rg, g.caca))

        #expect(MLX.all(MLX.isFinite(r.xFinal["bb_ca"]!)).item(Bool.self))
        // Two different noise draws will not land in the same place, so Rg is only
        // bounded. CA-CA is not: consecutive spacing along a real backbone is fixed
        // by chemistry, so it is the sharpest available check that what came out is
        // a polymer and not a point cloud. A port that drops connectivity lands
        // around 1.0 nm here and fails.
        #expect(g.rg < 1.2, "Rg \(g.rg) nm — upstream reaches 0.718")
        #expect(g.caca > 0.33 && g.caca < 0.43,
                "CA-CA \(g.caca) nm — a protein backbone is ~0.38 nm")
    }
}
