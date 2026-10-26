import Testing
@testable import RFD4Core

/// Ported from upstream `tests/test_atom37_encoding_alignment.py`.
///
/// Upstream docstring: "This test ensures that UNIFIED_ATOM37_ENCODING is correctly
/// structured and that collate.py derives its atom→slot mappings correctly from the
/// single source of truth."
///
/// Divergences from upstream, all deliberate:
///   - Upstream builds `biotite.structure.AtomArray`; this uses `RFD4Core.AtomArray`.
///   - Upstream's round-trip tests assert on biotite annotations that exist only
///     because biotite requires them (`occupancy`, `b_factor`). Those are carried but
///     not asserted, since nothing downstream in this port reads them.
///   - Random coordinates are replaced with a deterministic ramp. Upstream uses
///     `np.random.randn` with no seed; the assertions never depend on the values, and
///     an unseeded generator in a parity harness is a trap worth not importing.
@Suite("atom37 encoding alignment")
struct Atom37EncodingAlignmentTests {
    let vocabulary = Vocabulary.unifiedAtom37

    private func ramp(_ n: Int) -> [Vec3] {
        (0..<n).map { Vec3(Float($0), Float($0) * 0.5, Float($0) * -0.25) }
    }

    // MARK: test_vocabulary_slots_ignore_chain_classification

    @Test("vocabulary slots ignore chain classification",
          arguments: ["TRP", "NAG", "SEP", "DA", "A"],
          [ChainType.nonPolymer, ChainType.polypeptideL])
    func slotsIgnoreChainClassification(resName: String, chainType: ChainType) throws {
        let layout = try #require(vocabulary.atoms(of: resName))
        let occupied = layout.indices.filter { !layout[$0].isEmpty }

        var array = AtomArray(
            resName: [String](repeating: resName, count: occupied.count),
            atomName: occupied.map { layout[$0] },
            coord: ramp(occupied.count)
        )
        array.chainType = [ChainType](repeating: chainType, count: occupied.count)
        array.atomize = [Bool](repeating: false, count: occupied.count)

        #expect(array.atom37Slots(vocabulary) == occupied)

        // Explicit atomization overrides the residue layout: every atom goes to slot 1.
        array.atomize = [Bool](repeating: true, count: occupied.count)
        #expect(array.atom37Slots(vocabulary) == [Int](repeating: 1, count: occupied.count))
    }

    // MARK: test_unified_atom37_slot_1_is_center_atom

    @Test("slot 1 holds the token center atom")
    func slot1IsCenterAtom() throws {
        // Proteins: CA at slot 1, GLY and UNK included.
        for name in Vocabulary.standardAA.union([Vocabulary.unknownAA]) {
            let layout = try #require(vocabulary.atoms(of: name))
            #expect(layout[1].trimmingCharacters(in: .whitespaces) == "CA",
                    "\(name): expected CA at slot 1, got '\(layout[1])'")
        }

        // Nucleic acids: C1' at slot 1, including the unknown placeholders N and DN.
        for name in ["A", "C", "G", "U", "N", "DA", "DC", "DG", "DT", "DN"] {
            let layout = try #require(vocabulary.atoms(of: name))
            #expect(layout[1].trimmingCharacters(in: .whitespaces) == "C1'",
                    "\(name): expected C1' at slot 1, got '\(layout[1])'")
        }
    }

    // MARK: round trips

    private func roundTrip(
        resNames: [String], atomNames: [String], tokIdx: [Int],
        expectedSeq: [String], expectedResidues: [String], expectedAtoms: [String]
    ) throws {
        var array = AtomArray(resName: resNames, atomName: atomNames,
                              coord: ramp(resNames.count))
        array.tokIdx = tokIdx

        let encoded = try array.atom37Encoding(vocabulary)

        #expect(encoded.tokenCount == expectedSeq.count)
        #expect(encoded.xyz.allSatisfy { $0.count == Vocabulary.slotCount })
        #expect(encoded.mask.allSatisfy { $0.count == Vocabulary.slotCount })
        #expect(encoded.seq == expectedSeq.map { vocabulary.index(of: $0)! })

        let decoded = try AtomArray.from(encoded, vocabulary: vocabulary)

        #expect(decoded.count > 0, "decoded atom array should not be empty")
        #expect(Set(decoded.resName).sorted() == expectedResidues.sorted())
        for atom in expectedAtoms {
            #expect(decoded.atomName.contains(atom), "\(atom) atoms should be present")
        }
        // Every supplied atom survives the trip — upstream only checks presence of a
        // few names, but nothing is dropped, so assert the stronger property.
        #expect(decoded.count == resNames.count)
    }

    @Test("round trip: protein")
    func roundTripProtein() throws {
        try roundTrip(
            resNames: ["ALA", "ALA", "ALA", "GLY", "GLY", "GLY", "CYS", "CYS", "CYS", "CYS"],
            atomNames: ["N", "CA", "C", "N", "CA", "C", "N", "CA", "C", "SG"],
            tokIdx: [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            expectedSeq: ["ALA", "GLY", "CYS"],
            expectedResidues: ["ALA", "CYS", "GLY"],
            expectedAtoms: ["CA", "N", "C"]
        )
    }

    @Test("round trip: nucleic acid")
    func roundTripNucleicAcid() throws {
        try roundTrip(
            resNames: ["A", "A", "A", "C", "C", "C", "G", "G", "G"],
            atomNames: ["C1'", "C2'", "P", "C1'", "C2'", "P", "C1'", "C2'", "P"],
            tokIdx: [0, 0, 0, 1, 1, 1, 2, 2, 2],
            expectedSeq: ["A", "C", "G"],
            expectedResidues: ["A", "C", "G"],
            expectedAtoms: ["C1'", "P"]
        )
    }

    @Test("round trip: mixed protein and nucleic acid")
    func roundTripMixed() throws {
        try roundTrip(
            resNames: ["ALA", "ALA", "ALA", "DA", "DA", "DA", "GLY", "GLY", "GLY"],
            atomNames: ["N", "CA", "C", "C1'", "C2'", "P", "N", "CA", "C"],
            tokIdx: [0, 0, 0, 1, 1, 1, 2, 2, 2],
            expectedSeq: ["ALA", "DA", "GLY"],
            expectedResidues: ["ALA", "DA", "GLY"],
            expectedAtoms: ["CA", "C1'"]
        )
    }

    // MARK: vocabulary integrity (not upstream; guards the transcription)

    @Test("vocabulary transcription matches the checkpoint contract")
    func vocabularyIntegrity() {
        #expect(vocabulary.tokenCount == 147)
        #expect(vocabulary.ccdSnapshot == "2026-01-06")
        #expect(vocabulary.index(of: "<M>") == 0, "the mask token must be class 0")

        let atomMask = vocabulary.residueAtomMask()
        #expect(atomMask.count == 147)
        #expect(atomMask.allSatisfy { $0.count == 37 })

        let allowed = vocabulary.allowedResidueTypeMask()
        #expect(allowed.count == TokenType.count)
        #expect(allowed.allSatisfy { $0.count == 147 })
        // Row 4 is the union of every other row, so it is at least as permissive.
        for i in 0..<147 {
            let anyOther = (0..<TokenType.count)
                .contains { $0 != TokenType.unknown && allowed[$0][i] }
            #expect(allowed[TokenType.unknown][i] == anyOther)
        }
        #expect(allowed[TokenType.protein][vocabulary.index(of: "ALA")!])
        #expect(!allowed[TokenType.protein][vocabulary.index(of: "NAG")!])
        #expect(allowed[TokenType.glycan][vocabulary.index(of: "NAG")!])
    }
}
