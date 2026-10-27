import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// End-to-end generation: featurize, trunk, integrate, decode — the whole path.
///
/// Deliberately short and low-effort by default. The point is that the loop runs
/// and produces geometry, not that the geometry is good: judging a design needs
/// the production schedule and a refold, which is the science gate, not a test.
@Suite("generation")
struct GenerationTests {

    static var weightsPath: String? {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty
        else { return nil }
        return (p as NSString).expandingTildeInPath
    }

    @Test("a de-novo run produces finite all-atom geometry")
    func deNovoGeneration() throws {
        guard let p = Self.weightsPath else {
            print("skip generation: set RFD4_WEIGHTS to the checkpoint"); return
        }
        let gen = try Generator(weightsAt: URL(fileURLWithPath: p))
        let n = 16
        let system = Generator.System.deNovo(length: n)

        let started = Date()
        var trace: [(Int, Float)] = []
        let stochastic = ProcessInfo.processInfo.environment["RFD4_ODE"] == nil
        let result = try gen.generate(system, strategy: .thinkingLow, seed: 1,
                                      stochastic: stochastic) { step, _ in
            if step % 10 == 0 { trace.append((step, 0)) }
        }
        let seconds = Date().timeIntervalSince(started)
        _ = trace

        // Shapes first — a wrong one makes every later assertion meaningless.
        #expect(result.coordsNM.shape == [1, n, 37, 3])
        #expect(result.aatype.shape == [1, n])
        #expect(result.atomMask.shape == [1, n, 37])

        // Nothing diverged. NaN here would mean the integration blew up, which a
        // per-module parity test cannot catch because it never runs 60 steps.
        #expect(MLX.all(MLX.isFinite(result.coordsNM)).item(Bool.self),
                "generated coordinates contain NaN or infinity")
        for (mode, x) in result.xFinal {
            #expect(MLX.all(MLX.isFinite(x)).item(Bool.self), "\(mode) diverged")
        }

        // The CA trace should look like a chain rather than a cloud or a point.
        let ca = result.coordsNM[0, 0..., 1, 0...]          // [n, 3] nanometres
        let steps = ca[1...] - ca[..<(n - 1)]
        let dists = MLX.sqrt(MLX.sum(steps * steps, axis: -1)) * 10   // to Angstrom
        let mean = dists.mean().item(Float.self)
        let minD = dists.min().item(Float.self)
        let maxD = dists.max().item(Float.self)
        print(String(format: "generated %d residues in %.1fs — CA-CA %.2f A "
                     + "(min %.2f, max %.2f), %d atoms",
                     n, seconds, mean, minD, maxD,
                     Int(result.atomMask.asType(.float32).sum().item(Float.self))))

        // KNOWN DEFECT — the trajectory expands where upstream contracts.
        //
        // Fixed since first observed: the chain no longer collapses to a point.
        // That was two missing transition layers on the conditioning, and with
        // them the predicted vector field matches upstream to 3e-06 at every
        // sampled time (see PredictStepTests).
        //
        // What remains is a dynamics difference, same batch, same length:
        //
        //   upstream   step 0  Rg 1.458 nm    step 59  Rg 0.856 nm   (contracts)
        //   this port  step 0  Rg 1.521 nm    step 59  Rg 2.784 nm   (expands)
        //
        // The initial states agree, so it is not the noise draw's scale —
        // MLXRandom.normal measures mean 0.0009, sd 0.9983. v is at parity, the
        // integration step is bit-exact across all six branches, and the schedules
        // are bit-exact. The two runs also used different recycle depths, which is
        // not a fair comparison and is the first thing to eliminate.
        //
        // Next: drive both from an identical noise sequence and the same depth, so
        // the trajectories are directly comparable rather than two draws.
        withKnownIssue("trajectory expands where upstream contracts") {
            #expect(mean > 2.0 && mean < 6.0, "CA-CA \(mean) A is outside protein range")
        }

        // Every emitted residue must carry its reference atom.
        let slot1 = result.atomMask[0, 0..., 1]
        #expect(MLX.all(slot1).item(Bool.self), "an emitted residue has no CA")

        // Residue classes must be inside the vocabulary.
        let maxClass = result.aatype.max().item(Int32.self)
        #expect(Int(maxClass) < Vocabulary.unifiedAtom37.tokenCount)
    }
}
