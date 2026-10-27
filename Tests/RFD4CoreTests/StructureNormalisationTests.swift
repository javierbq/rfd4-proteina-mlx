import Foundation
import RFD4Core
import Testing

/// The two parse-time transforms a design input needs, and what goes wrong without
/// them.
///
/// Both were found by review rather than by the parity suite, because the shipped
/// PD-L1 target happens to carry neither an alternate conformer nor a
/// selenomethionine. Every measurement in this port was taken on that one file, so
/// "it matches upstream" said nothing about either.
@Suite("structure normalisation")
struct StructureNormalisationTests {

    static func array(_ rows: [(res: String, atom: String, chain: String, rid: Int,
                                alt: String, x: Float)]) -> AtomArray {
        var a = AtomArray(resName: rows.map(\.res), atomName: rows.map(\.atom),
                          coord: rows.map { Vec3($0.x, 0, 0) })
        a.chainID = rows.map(\.chain)
        a.resID = rows.map(\.rid)
        a.altLocID = rows.map(\.alt)
        // Real elements, not the atom name's first letter: selenium is "SE", and a
        // fixture that already called it "S" would make the conversion look like a
        // no-op.
        a.element = rows.map { $0.atom == "SE" ? "SE" : String($0.atom.prefix(1)) }
        return a
    }

    // MARK: - alternate conformers

    /// One atom per (chain, residue, name), whatever the file models.
    ///
    /// Measured against upstream on the shipped target with a two-conformer TYR56
    /// OH: it collapsed to one atom and kept conformer `A`, even though `B` carried
    /// the higher occupancy (0.60 against 0.40).
    @Test("alternate conformers collapse to one atom, keeping the first")
    func altlocsCollapse() {
        let a = Self.array([
            ("TYR", "CA", "A", 56, ".", 0),
            ("TYR", "OH", "A", 56, "A", 8.539),
            ("TYR", "OH", "A", 56, "B", 10.039),
            ("THR", "CA", "A", 57, ".", 20),
        ])
        let collapsed = a.collapsingAltlocs()
        #expect(collapsed.count == 3, "expected one OH, got \(collapsed.count) atoms")
        let oh = collapsed.indices.filter { collapsed.atomName[$0] == "OH" }
        #expect(oh.count == 1)
        #expect(collapsed.coord[oh[0]].x == 8.539, "kept B where upstream kept A")

        // An unlabelled row beats a labelled one regardless of file order, so the
        // choice does not depend on how the file happens to be written.
        let blankLast = Self.array([
            ("TYR", "OH", "A", 56, "B", 1),
            ("TYR", "OH", "A", 56, ".", 2),
        ])
        let c2 = blankLast.collapsingAltlocs()
        #expect(c2.count == 1 && c2.coord[0].x == 2)

        // Same atom name in a different residue, or a different chain, is a
        // different atom and must survive.
        let distinct = Self.array([
            ("TYR", "OH", "A", 56, "A", 1),
            ("TYR", "OH", "A", 57, "A", 2),
            ("TYR", "OH", "B", 56, "A", 3),
        ])
        #expect(distinct.collapsingAltlocs().count == 3)

        // A file with no altlocs is returned untouched.
        let plain = Self.array([("GLY", "CA", "A", 1, ".", 0), ("GLY", "N", "A", 1, "?", 1)])
        #expect(plain.collapsingAltlocs().count == 2)
    }

    /// What collapsing prevents: a duplicate row inflates a group past its distinct
    /// atom count and double-counts a hotspot.
    @Test("uncollapsed conformers double-count an atom")
    func altlocsWouldDoubleCount() {
        let a = Self.array([
            ("TYR", "OH", "A", 56, "A", 1),
            ("TYR", "OH", "A", 56, "B", 2),
        ])
        #expect(a.count == 2, "the premise of the collapse no longer holds")
        #expect(a.collapsingAltlocs().count == 1)
    }

    // MARK: - selenomethionine

    /// `convert_mse_to_met`: MSE is read as MET, `SE` as `SD`, element as `S`.
    ///
    /// Selenomethionine is how a crystallographer solves the phase problem, so it is
    /// in a large share of RCSB entries and is not in the canonical-20 alphabet.
    /// Left as MSE, every atom fails the vocabulary lookup, the residue drops out of
    /// the chain, and its neighbours become adjacent — a shorter layout and a
    /// peptide bond asserted across a gap the structure does not have.
    @Test("MSE is read as MET with SE renamed to SD")
    func mseBecomesMet() {
        let a = Self.array([
            ("MSE", "N", "A", 36, ".", 0),
            ("MSE", "CA", "A", 36, ".", 1),
            ("MSE", "SE", "A", 36, ".", 2),
            ("MET", "SE", "A", 37, ".", 3),   // not MSE: must be left alone
        ])
        let converted = a.convertingMSEToMET()
        #expect(converted.resName == ["MET", "MET", "MET", "MET"])
        #expect(converted.atomName == ["N", "CA", "SD", "SE"],
                "SE was renamed outside an MSE residue, or not renamed inside one")
        #expect(converted.element?[2] == "S", "the MSE selenium kept element SE")
        #expect(converted.element?[3] == "SE",
                "residue 37 is a real MET with a selenium; its element must be left alone")
        #expect(converted.count == a.count, "conversion must not drop or add atoms")

        // The whole point: after conversion the atoms resolve to atom37 slots.
        let vocab = Vocabulary.unifiedAtom37
        let before = a.atom37Slots(vocab)
        let after = converted.atom37Slots(vocab)
        #expect(before[0 ..< 3].allSatisfy { $0 < 0 },
                "MSE already resolved — this test's premise is gone")
        #expect(after[0 ..< 3].allSatisfy { $0 >= 0 },
                "MET atoms still do not resolve after conversion")
    }

    @Test("normalizedForDesign applies both")
    func bothTogether() {
        let a = Self.array([
            ("MSE", "SE", "A", 36, "A", 1),
            ("MSE", "SE", "A", 36, "B", 2),
            ("MSE", "CA", "A", 36, ".", 3),
        ])
        let n = a.normalizedForDesign()
        #expect(n.count == 2)
        #expect(n.resName.allSatisfy { $0 == "MET" })
        #expect(n.atomName.contains("SD") && !n.atomName.contains("SE"))
    }
}
