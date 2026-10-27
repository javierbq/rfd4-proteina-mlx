import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// Ported from upstream `tests/pipeline/test_bond_path_lengths.py` and
/// `test_canonical_polymer_ordering.py`.
///
/// Those tests drive a bond-graph BFS over an AtomArray, which this port does not
/// implement — a contig gives a linear polymer per chain run, so the hop count is
/// the index separation. What they establish and this inherits is the **contract**:
/// what the values mean, which pairs are connected, and how "no path" is told apart
/// from "too far".
@Suite("bond path lengths")
struct BondPathLengthTests {

    /// A synthetic structure with every residue of chain A present, 1...500.
    static func layout(_ contig: String, draw: Int = 10) throws -> TokenLayout {
        try TokenLayout.make(
            contig: contig,
            residuesInChain: { $0 == "A" ? Array(1 ... 500) : [] },
            draw: { lo, hi in lo == hi ? lo : draw })
    }

    /// `"Connected paths beyond the cutoff get overflow; disconnected pairs remain
    /// zero."` — upstream's `test_no_path_and_too_far_are_told_apart`.
    ///
    /// The two are different facts that share a feature, and the distinction is the
    /// whole reason the cutoff sentinel exists. Collapsing them loses the difference
    /// between "far along this chain" and "not this chain at all".
    @Test("no path and too far are told apart")
    func noPathVersusTooFar() throws {
        // One chain of 200, so pairs exist both inside and beyond the 65 cutoff.
        let one = try Self.layout("200")
        let path = one.tokenBondPathLength()
        #expect(path[0][1] == 1)
        #expect(path[0][64] == 64)
        #expect(path[0][65] == 65, "the last exactly-reported hop count")
        #expect(path[0][150] == 65, "beyond the cutoff must read as the sentinel")
        #expect(path[0][0] == 0, "the self-diagonal is 0")

        // Two chains: every cross pair is disconnected, which is the SAME value as
        // the diagonal. Upstream says so explicitly — 0 means self, disconnected or
        // cross-group alike.
        let two = try Self.layout("40, /0, A17-50")
        let cross = two.tokenBondPathLength()
        let breakAt = try #require(two.chainBreaks().first)
        #expect(cross[breakAt - 1][breakAt] == 0, "a chain break must read as disconnected")
        #expect(cross[0][two.count - 1] == 0)
        // …and within each chain it still counts hops.
        #expect(cross[0][5] == 5)
        #expect(cross[breakAt][breakAt + 5] == 5)
    }

    /// Ordering marks only truly adjacent residues, and carries direction.
    ///
    /// Upstream's gap fixture pins the negative case: residues 3 and 7 are bridged
    /// through filler for path length, but stay `0` in the ordering because the input
    /// does not imply they are adjacent. Here the equivalent is a chain break.
    @Test("ordering is signed and stops at a chain break")
    func orderingStopsAtBreak() throws {
        let two = try Self.layout("40, /0, A17-50")
        let o = two.canonicalPolymerOrdering()
        let breakAt = try #require(two.chainBreaks().first)

        // +1 at [downstream, upstream], -1 at the transpose.
        #expect(o[1][0] == 1)
        #expect(o[0][1] == -1)
        #expect(o[2][0] == 0, "ordering must be adjacency, not reachability")
        // Across the break: nothing, in either direction.
        #expect(o[breakAt][breakAt - 1] == 0, "the chain break was bonded")
        #expect(o[breakAt - 1][breakAt] == 0)
        // Immediately after the break the next chain picks up again.
        #expect(o[breakAt + 1][breakAt] == 1)

        // Exactly 2 × (tokens - chains) non-zero entries.
        let nonZero = o.flatMap { $0 }.filter { $0 != 0 }.count
        #expect(nonZero == 2 * (two.count - two.chainCount))
    }

    /// Degenerate layouts must not trap, and must not invent connectivity.
    @Test("empty, single-token and single-chain layouts")
    func degenerate() throws {
        let empty = TokenLayout(tokens: [])
        #expect(empty.count == 0)
        #expect(empty.chainBreaks().isEmpty)
        #expect(empty.chainRanges().isEmpty)
        #expect(empty.tokenBondPathLength().isEmpty)
        #expect(empty.canonicalPolymerOrdering().isEmpty)

        let one = TokenLayout(tokens: [
            .init(chainIndex: 0, chainID: "A", segmentIndex: 0, inputResidue: nil),
        ])
        #expect(one.count == 1)
        #expect(one.chainBreaks().isEmpty)
        #expect(one.chainRanges() == [0 ..< 1])
        #expect(one.tokenBondPathLength() == [[0]])
        #expect(one.canonicalPolymerOrdering() == [[0]])

        let chain = try Self.layout("A17-50")
        #expect(chain.chainCount == 1)
        #expect(chain.chainBreaks().isEmpty)
        #expect(chain.chainRanges() == [0 ..< chain.count])
    }

    /// Three chains, so the two-chain case cannot pass by accident.
    @Test("three chains partition cleanly")
    func threeChains() throws {
        let three = try Self.layout("30, /0, A17-40, /0, 15")
        #expect(three.chainCount == 3)
        #expect(three.chainBreaks() == [30, 30 + 24])
        #expect(three.chainRanges() == [0 ..< 30, 30 ..< 54, 54 ..< 69])
        let groups = three.chainGroupIDs()
        #expect(Set(groups).count == 3)
        let path = three.tokenBondPathLength()
        // Every cross-chain pair, in both directions.
        for a in three.chainRanges()[0] {
            for b in three.chainRanges()[2] {
                #expect(path[a][b] == 0 && path[b][a] == 0)
            }
        }
    }

    /// A contig that asks for a bond the input cannot support must be refused.
    ///
    /// Upstream bonds tokens consecutively within a chain run whatever their input
    /// numbering, then rejects the result:
    ///
    ///     [bond_length] A_1:THR4:C -- A_1:VAL5:N = 24.61 Å (limit 2.0 Å).
    ///
    /// Measured by running `"A17-20, A30-33"` through its own pipeline. Without the
    /// check this port accepts it and designs against a chain carrying a 24 Å peptide
    /// bond — every feature well-formed, only the chemistry impossible.
    @Test("a contig that bonds across a gap is refused, as upstream refuses it")
    func impossibleBondRefused() throws {
        let layout = try Self.layout("A17-20, A30-33")
        #expect(layout.count == 8)
        #expect(layout.chainCount == 1, "one run, so upstream would bond these")

        let slots = Vocabulary.slotCount
        var coords = [[Vec3?]](repeating: [Vec3?](repeating: nil, count: slots), count: 8)
        // Four residues 3.8 Å apart, then a 24 Å jump, then four more.
        for t in 0 ..< 8 {
            let x = t < 4 ? Float(t) * 3.8 : Float(t) * 3.8 + 24
            coords[t][0] = Vec3(x, 0, 0)            // N
            coords[t][1] = Vec3(x + 1.2, 0, 0)      // CA
            coords[t][2] = Vec3(x + 2.4, 0, 0)      // C
        }
        let known = [Int?](repeating: 1, count: 8)
        let bad = DesignProblem(layout: layout, coords: coords, knownSequence: known)
        #expect(throws: DesignProblemError.self) { try bad.validate() }
        #expect(throws: DesignProblemError.self) { try bad.validatedSystem() }

        // `/0` is not the escape here: two runs both claiming chain A is itself
        // rejected, by upstream and by this port alike. The honest fix is to crop
        // the input, which is what upstream's own message asks for ("preserve
        // missing-residue gaps instead of bonding across them").
        #expect(throws: ContigError.self) { try Self.layout("A17-20, /0, A30-33") }

        // A contiguous range of the same residues bonds at real peptide distance and
        // passes — so the check is testing the geometry, not merely refusing
        // everything with two preserved segments.
        let contiguous = try Self.layout("A17-24")
        var fine = [[Vec3?]](repeating: [Vec3?](repeating: nil, count: slots), count: 8)
        for t in 0 ..< 8 {
            let x = Float(t) * 3.8
            fine[t][0] = Vec3(x, 0, 0)              // N
            fine[t][2] = Vec3(x + 2.45, 0, 0)       // C, 1.35 A from the next N
        }
        let good = DesignProblem(layout: contiguous, coords: fine, knownSequence: known)
        #expect(throws: Never.self) { try good.validate() }
    }

    /// A preserved segment is as long as the residues that exist, not as its range.
    ///
    /// Upstream selects `chain_id == c & start <= res_id <= end` and counts what it
    /// finds (`contig_spec.py:_build_run_preserved_atoms`). Crystal structures have
    /// unmodelled loops as a matter of course, so `end - start + 1` is the wrong
    /// length for an ordinary input — and wrong silently: the surplus tokens carry no
    /// atoms, so they enter the batch with an all-zero atom37 mask, NaN coordinates
    /// and NaN sequence probabilities, shifting the chain break and every
    /// conditioning array after them.
    @Test("a segment is the residues present, not the span of its range")
    func unmodelledLoop() throws {
        // Chain A runs 17-131 but residues 50-55 were never modelled.
        let present = Array(17 ... 49) + Array(56 ... 131)
        let source: (String) -> [Int] = { $0 == "A" ? present : [] }

        let layout = try TokenLayout.make(
            contig: "20, /0, A17-131", residuesInChain: source, draw: { lo, _ in lo })
        #expect(layout.count == 20 + present.count, "ghost tokens for the unmodelled loop")
        #expect(layout.count == 129, "20 designed + 109 present, not 20 + 115")
        #expect(layout.chainBreaks() == [20])

        // And every preserved token points at a residue that is really there — the
        // arithmetic `start + offset` form would aim everything past the hole six
        // residues short, which no count check would catch.
        let preserved = layout.tokens.compactMap(\.inputResidue)
        #expect(preserved == present)
        #expect(!preserved.contains(50), "token points at an unmodelled residue")
        #expect(preserved.contains(49) && preserved.contains(56))

        // Asking for residues the chain does not have at all is a typo, and upstream
        // raises rather than clipping.
        #expect(throws: ContigError.self) {
            try TokenLayout.make(contig: "A17-200", residuesInChain: source,
                                 draw: { lo, _ in lo })
        }
        #expect(throws: ContigError.self) {
            try TokenLayout.make(contig: "A1-30", residuesInChain: source,
                                 draw: { lo, _ in lo })
        }
        // A range entirely inside the hole matches nothing.
        #expect(throws: ContigError.self) {
            try TokenLayout.make(contig: "A50-55", residuesInChain: source,
                                 draw: { lo, _ in lo })
        }
        // A chain the file does not contain.
        #expect(throws: ContigError.self) {
            try TokenLayout.make(contig: "B17-30", residuesInChain: source,
                                 draw: { lo, _ in lo })
        }
    }

    /// A designed token has no coordinates, so it cannot be validated — and must not
    /// be mistaken for a violation.
    @Test("designed segments are not bond-checked")
    func designedSegmentsSkipped() throws {
        let layout = try Self.layout("4, /0, A17-20")
        let slots = Vocabulary.slotCount
        var coords = [[Vec3?]](repeating: [Vec3?](repeating: nil, count: slots),
                               count: layout.count)
        // Only the preserved half carries coordinates, and they are far apart — but
        // they are in a different chain from the designed tokens, so nothing bonds.
        for t in 4 ..< layout.count {
            let x = Float(t) * 3.8
            coords[t][0] = Vec3(x, 0, 0)
            coords[t][2] = Vec3(x + 2.4, 0, 0)
        }
        let problem = DesignProblem(layout: layout, coords: coords,
                                    knownSequence: [Int?](repeating: nil, count: layout.count))
        #expect(throws: Never.self) { try problem.validate() }
    }
}
