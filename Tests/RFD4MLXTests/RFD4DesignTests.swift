import Foundation
import RFD4Core
import Testing
@testable import RFD4MLX

/// The host-facing design path: the size model, the frame recovery, and the layout
/// decisions a host would otherwise have to make for itself.
///
/// Everything here but the last test runs without weights, which is the point —
/// the parts that decide what the model is ASKED are separable from the model, and they
/// are where a silent wrong answer comes from. A contig that bonds across a gap, a
/// hotspot declared on the wrong atom and a design placed by a bad rotation all produce
/// well-formed output.
@Suite("rfd4 design")
struct RFD4DesignTests {

    // MARK: - Budget

    /// The four points `validation/benchmark/peak-memory.md` measured, against the fit
    /// that was taken from them. Pinned here rather than only in prose because a host's
    /// refusal is computed from this and nothing else.
    @Test("the peak-memory fit reproduces its own measurements")
    func budgetMatchesMeasurements() {
        let measured: [(n: Int, fp32: Double, fp16: Double)] = [
            (256, 3.196, 1.598),
            (384, 5.528, 2.764),
            (512, 8.936, 4.468),
            (638, 13.240, 6.620),
        ]
        for point in measured {
            let fp32 = Double(RFD4Budget.predictedPeakBytes(tokens: point.n,
                                                            precision: .fp32)) / 1e9
            let fp16 = Double(RFD4Budget.predictedPeakBytes(tokens: point.n,
                                                            precision: .fp16)) / 1e9
            // 16 MB is the fit's own stated maximum error.
            #expect(abs(fp32 - point.fp32) < 0.016,
                    "fp32 n=\(point.n): \(fp32) vs \(point.fp32)")
            #expect(abs(fp16 - point.fp16) < 0.016,
                    "fp16 n=\(point.n): \(fp16) vs \(point.fp16)")
        }
    }

    /// The albumin campaign is the only END-TO-END number there is — a real generation
    /// with real weights, not the bench the coefficients came from. If the fit ever
    /// stops predicting it, the guard built on the fit is predicting the wrong thing.
    @Test("the fit predicts the albumin campaign's measured peak")
    func budgetMatchesCampaign() {
        // 100 designs at 210-230 tokens, fp32, high-water mark 2.79 GB.
        let predicted = Double(RFD4Budget.predictedPeakBytes(tokens: 230,
                                                             precision: .fp32)) / 1e9
        #expect(predicted >= 2.79, "a guard must not under-predict: \(predicted)")
        #expect(predicted < 2.79 * 1.05, "and not by much either: \(predicted)")
    }

    /// Half precision is half the memory, which is also the check that nothing in the
    /// trunk is still float32 — the measurement the ratio was taken from would move off
    /// 2.000 the moment it was.
    @Test("fp16 is half of fp32")
    func budgetPrecisionRatio() {
        for n in [128, 256, 512, 700] {
            let fp32 = Double(RFD4Budget.predictedPeakBytes(tokens: n, precision: .fp32))
            let fp16 = Double(RFD4Budget.predictedPeakBytes(tokens: n, precision: .fp16))
            #expect(abs(fp32 / fp16 - 2.0) < 0.01, "n=\(n): ratio \(fp32 / fp16)")
        }
    }

    /// `maxTokens` and `predictedPeakBytes` are the same arithmetic in opposite
    /// directions, so the boundary must agree from both sides. It is a REFUSAL boundary:
    /// one token of disagreement means the guard's own advice names a size it refuses.
    @Test("maxTokens is the exact boundary predictedPeakBytes describes")
    func budgetRoundTrip() {
        for budgetGB in [2.0, 4.0, 8.0, 18.0, 36.0] {
            let budget = Int(budgetGB * 1e9)
            for precision in [RFD4Budget.Precision.fp16, .fp32] {
                let n = RFD4Budget.maxTokens(budgetBytes: budget, precision: precision)
                #expect(n > 0)
                #expect(RFD4Budget.predictedPeakBytes(tokens: n, precision: precision)
                        <= budget, "\(precision) \(budgetGB) GB: n=\(n) does not fit")
                #expect(RFD4Budget.predictedPeakBytes(tokens: n + 1, precision: precision)
                        > budget, "\(precision) \(budgetGB) GB: n=\(n + 1) also fits")
            }
        }
    }

    @Test("an impossible budget yields no tokens rather than a negative count")
    func budgetTooSmall() {
        #expect(RFD4Budget.maxTokens(budgetBytes: 0) == 0)
        #expect(RFD4Budget.maxTokens(budgetBytes: -1) == 0)
        #expect(RFD4Budget.predictedPeakBytes(tokens: 0) >= 0)
        #expect(RFD4Budget.predictedPeakBytes(tokens: -5) >= 0)
    }

    // MARK: - Superposition

    /// The solver recovers a known rigid motion. This is the one routine whose failure
    /// is invisible downstream: a bad rotation gives a well-formed complex in the wrong
    /// pose, and nothing after it can tell.
    @Test("superposition recovers a known rotation exactly")
    func superpositionSelfTest() {
        let residual = Superposition.selfTestResidual()
        #expect(residual < 1e-10, "residual \(residual)")
    }

    @Test("a perfect fit reports zero rather than NaN")
    func superpositionPerfectFit() {
        let points: [SIMD3<Double>] = [
            SIMD3(1, 2, 3), SIMD3(4, 0, -1), SIMD3(-2, 5, 2), SIMD3(0, 0, 7),
        ]
        let fit = Superposition.fit(mobile: points, reference: points)
        #expect(fit != nil)
        #expect(fit!.rmsd.isFinite)
        #expect(fit!.rmsd < 1e-10)
    }

    @Test("fewer than three points is refused rather than guessed")
    func superpositionNeedsThreePoints() {
        let two: [SIMD3<Double>] = [SIMD3(0, 0, 0), SIMD3(1, 0, 0)]
        #expect(Superposition.fit(mobile: two, reference: two) == nil)
        #expect(Superposition.fit(mobile: two, reference: [SIMD3(0, 0, 0)]) == nil)
    }

    // MARK: - Layout

    /// A straight chain of residues whose C(i)-N(i+1) distances are peptide-bond length.
    /// Not a protein; it is the smallest thing that makes the bond test answer `true`.
    static func chain(_ count: Int, startingAt x0: Float = 0, name: String = "ALA")
        -> [RFD4Design.TargetResidue] {
        (0 ..< count).map { i in
            let x = x0 + Float(i) * 3.8
            return RFD4Design.TargetResidue(name: name, atoms: [
                RFD4Design.Atom(name: "N", xyz: Vec3(x - 1.2, 0, 0)),
                RFD4Design.Atom(name: "CA", xyz: Vec3(x, 0, 0)),
                RFD4Design.Atom(name: "C", xyz: Vec3(x + 1.2, 0, 0)),
                RFD4Design.Atom(name: "O", xyz: Vec3(x + 1.6, 1.0, 0)),
                RFD4Design.Atom(name: "CB", xyz: Vec3(x, 1.5, 0)),
            ])
        }
    }

    @Test("a continuous backbone is one run")
    func runsContiguous() {
        let runs = RFD4Design.runs(in: Self.chain(10))
        #expect(runs == [0 ..< 10])
    }

    /// The case that is wrong by default: a gap in the backbone must become a chain
    /// break, or the contig asks for a peptide bond across it and `validate()` refuses.
    @Test("a gap in the backbone splits the target")
    func runsSplitAtGap() {
        var target = Self.chain(5)
        target += Self.chain(4, startingAt: 100)
        let runs = RFD4Design.runs(in: target)
        #expect(runs == [0 ..< 5, 5 ..< 9])
    }

    /// A residue that cannot evidence a bond does not get one. The alternative is to
    /// assume connectivity the input never showed.
    @Test("a missing backbone atom reads as a break, not as a bond")
    func runsBreakOnMissingAtom() {
        var target = Self.chain(4)
        // Residue 1 loses its C, so nothing joins it to residue 2.
        target[1] = RFD4Design.TargetResidue(
            name: target[1].name,
            atoms: target[1].atoms.filter { $0.name != "C" })
        #expect(RFD4Design.runs(in: target) == [0 ..< 2, 2 ..< 4])
    }

    @Test("the contig puts the design first and each target segment on its own chain")
    func contigShape() {
        #expect(RFD4Design.contig(length: 60, segments: [0 ..< 10])
                == "60, /0, B1-10")
        #expect(RFD4Design.contig(length: 60, segments: [0 ..< 10, 10 ..< 25])
                == "60, /0, B1-10, /0, C1-15")
    }

    /// The contig this builds must survive the real parser and produce the layout it
    /// claims — including the designed chain coming out as 'A', which is what makes the
    /// emitted complex read the way the CLI's does.
    @Test("a gapped target's contig realizes into the expected layout")
    func gappedContigRealizes() throws {
        var target = Self.chain(6)
        target += Self.chain(7, startingAt: 100)
        let segments = RFD4Design.runs(in: target)
        let contig = RFD4Design.contig(length: 20, segments: segments)
        var available: [String: [Int]] = [:]
        for (index, segment) in segments.enumerated() {
            available[RFD4Design.targetChainIDs[index]] = Array(1 ... segment.count)
        }
        let layout = try TokenLayout.make(contig: contig,
                                          residuesInChain: { available[$0] ?? [] },
                                          draw: { lo, _ in lo })
        #expect(layout.count == 20 + 13)
        #expect(layout.designedMask().prefix(20).allSatisfy { $0 })
        #expect(layout.designedMask().dropFirst(20).allSatisfy { !$0 })
        // Three chains: the design, and one per target segment.
        #expect(layout.chainCount == 3)
        #expect(layout.tokens[0].chainID == "A")
        #expect(layout.tokens[20].chainID == "B")
        #expect(layout.tokens[26].chainID == "C")
    }

    /// Too many disjoint pieces is refused by name rather than running out of chain ids
    /// inside the contig parser, where the error would be about letters rather than
    /// about the selection.
    @Test("an impossibly fragmented target is refused with its own message")
    func tooManyRunsRefused() {
        var target: [RFD4Design.TargetResidue] = []
        for i in 0 ..< 30 { target += Self.chain(1, startingAt: Float(i) * 100) }
        #expect(throws: RFD4Design.Error.self) {
            try RFD4Design.preflight(target: target,
                                     options: RFD4Design.Options(designLength: 10))
        }
    }

    // MARK: - Hotspots

    /// A hotspot is a claim about which SURFACE to engage, so it is declared on the
    /// sidechain. Flagging the backbone too would put the feature on atoms that are
    /// buried in the chain rather than presented by it.
    @Test("a hotspot is declared on sidechain atoms, not backbone ones")
    func hotspotUsesSidechain() {
        let vocab = Vocabulary.unifiedAtom37
        let tyrosine = RFD4Design.TargetResidue(name: "TYR", atoms: [
            RFD4Design.Atom(name: "N", xyz: Vec3(0, 0, 0)),
            RFD4Design.Atom(name: "CA", xyz: Vec3(1, 0, 0)),
            RFD4Design.Atom(name: "C", xyz: Vec3(2, 0, 0)),
            RFD4Design.Atom(name: "O", xyz: Vec3(3, 0, 0)),
            RFD4Design.Atom(name: "CB", xyz: Vec3(1, 1, 0)),
            RFD4Design.Atom(name: "CG", xyz: Vec3(1, 2, 0)),
            RFD4Design.Atom(name: "OH", xyz: Vec3(1, 5, 0)),
        ])
        let slots = Set(RFD4Design.hotspotAtomSlots(residue: tyrosine, vocabulary: vocab))
        #expect(slots.count == 3)
        for name in ["CB", "CG", "OH"] {
            #expect(slots.contains(vocab.slot(residue: "TYR", atom: name)!),
                    "\(name) should be flagged")
        }
        for name in ["N", "CA", "C", "O"] {
            #expect(!slots.contains(vocab.slot(residue: "TYR", atom: name)!),
                    "\(name) should not be flagged")
        }
    }

    /// Glycine has no sidechain, so the residue-level reading has to fall back to
    /// something rather than contributing no hotspot at all.
    @Test("glycine falls back to CA")
    func hotspotGlycineFallsBackToCA() {
        let vocab = Vocabulary.unifiedAtom37
        let glycine = RFD4Design.TargetResidue(name: "GLY", atoms: [
            RFD4Design.Atom(name: "N", xyz: Vec3(0, 0, 0)),
            RFD4Design.Atom(name: "CA", xyz: Vec3(1, 0, 0)),
            RFD4Design.Atom(name: "C", xyz: Vec3(2, 0, 0)),
            RFD4Design.Atom(name: "O", xyz: Vec3(3, 0, 0)),
        ])
        #expect(RFD4Design.hotspotAtomSlots(residue: glycine, vocabulary: vocab)
                == [vocab.slot(residue: "GLY", atom: "CA")!])
    }

    /// An unresolved coordinate cannot carry a hotspot: the feature is gated by the
    /// atom37 mask, and a NaN would poison the rigid group's centroid besides.
    @Test("an unresolved sidechain atom is not flagged")
    func hotspotSkipsNaN() {
        let vocab = Vocabulary.unifiedAtom37
        let serine = RFD4Design.TargetResidue(name: "SER", atoms: [
            RFD4Design.Atom(name: "CA", xyz: Vec3(1, 0, 0)),
            RFD4Design.Atom(name: "CB", xyz: Vec3(.nan, .nan, .nan)),
        ])
        #expect(RFD4Design.hotspotAtomSlots(residue: serine, vocabulary: vocab)
                == [vocab.slot(residue: "SER", atom: "CA")!])
    }

    // MARK: - Refusals

    /// Each of these is an input the model would ACCEPT and quietly reinterpret, which
    /// is why they are refused here. All of them are reachable without a loaded model —
    /// that is the property a host relies on to refuse before an 871 MB download.
    @Test("inputs the model would quietly reinterpret are refused by name")
    func refusals() {
        let good = Self.chain(5)
        #expect(throws: RFD4Design.Error.self) {
            try RFD4Design.preflight(target: [],
                                     options: RFD4Design.Options(designLength: 10))
        }
        #expect(throws: RFD4Design.Error.self) {
            try RFD4Design.preflight(target: good,
                                     options: RFD4Design.Options(designLength: 0))
        }
        #expect(throws: RFD4Design.Error.self) {
            var options = RFD4Design.Options(designLength: 10)
            options.hotspots = [99]
            try RFD4Design.preflight(target: good, options: options)
        }
        // A residue the featurizer has no template for. Its own reader DROPS this and
        // returns success, shifting every hotspot position after it.
        var modified = good
        modified[2] = RFD4Design.TargetResidue(name: "MSE", atoms: modified[2].atoms)
        #expect(throws: RFD4Design.Error.self) {
            try RFD4Design.preflight(target: modified,
                                     options: RFD4Design.Options(designLength: 10))
        }
        // And the shape that is fine.
        #expect(throws: Never.self) {
            var options = RFD4Design.Options(designLength: 10)
            options.hotspots = [0, 4]
            try RFD4Design.preflight(target: good, options: options)
        }
    }

    @Test("token count is what the budget is denominated in")
    func tokenCount() {
        #expect(RFD4Design.tokenCount(target: Self.chain(150), designLength: 76) == 226)
    }

    // MARK: - End to end

    static var weightsPath: String? {
        guard let p = ProcessInfo.processInfo.environment["RFD4_WEIGHTS"], !p.isEmpty
        else { return nil }
        return (p as NSString).expandingTildeInPath
    }

    /// The whole path, including the frame recovery that no unit test can exercise:
    /// the model places the target itself, from a random rotation, so "the design is
    /// beside the target" is only true if the superposition is right.
    @Test("a design comes back in the caller's frame")
    func endToEnd() throws {
        guard let path = Self.weightsPath else {
            print("skip rfd4 design e2e: set RFD4_WEIGHTS"); return
        }
        let generator = try Generator(weightsAtPathOrPack: path)
        let target = Self.chain(24)
        var options = RFD4Design.Options(designLength: 12)
        options.hotspots = [8, 9, 10]
        options.strategy = .thinkingLow
        options.seed = 3

        let result = try RFD4Design.run(generator, target: target, options: options)
        #expect(result.designResidues.count == 12)
        #expect(result.designSequence.count == 12)
        #expect(result.tokenCount == 36)
        for residue in result.designResidues {
            for atom in residue {
                #expect(atom.xyz.x.isFinite && atom.xyz.y.isFinite && atom.xyz.z.isFinite)
            }
        }
        // The claim the superposition makes: the rebuilt target lands ON the supplied
        // one. A failed recovery puts it tens of Angstrom away, so this is a wide bound
        // that still separates "recovered" from "not".
        #expect(result.stats.targetRebuildRMSDA < 5.0,
                "rebuild RMSD \(result.stats.targetRebuildRMSDA)")
        // And the design is somewhere near the target rather than in another frame.
        // Unwrapped rather than defaulted: a nil here means nothing was measured, which
        // would pass any `?? 0` comparison while proving nothing.
        let interface = try #require(result.stats.interfaceMinA)
        #expect(interface < 50.0, "interface \(interface)")
        #expect(result.stats.designCACAMeanA > 3.0
                && result.stats.designCACAMeanA < 4.5,
                "CA-CA \(result.stats.designCACAMeanA)")
    }

    /// Cancellation is the only way to stop a run, and a design is minutes long, so a
    /// hook that silently does nothing is a Cancel button that does nothing.
    @Test("a cancelled run throws rather than finishing")
    func cancellation() throws {
        guard let path = Self.weightsPath else {
            print("skip rfd4 cancel: set RFD4_WEIGHTS"); return
        }
        let generator = try Generator(weightsAtPathOrPack: path)
        var options = RFD4Design.Options(designLength: 8)
        options.strategy = .thinkingLow
        // Cancel after the first step has completed.
        options.shouldCancel = { true }
        #expect(throws: GenerationCancelled.self) {
            _ = try RFD4Design.run(generator, target: Self.chain(12), options: options)
        }
    }
}
