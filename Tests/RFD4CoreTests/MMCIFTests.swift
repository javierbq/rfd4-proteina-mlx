import Foundation
import Testing
@testable import RFD4Core

/// The mmCIF reader, against biotite's own parse of the same files.
///
/// The fixtures are RFD4's shipped examples, not synthetic files, so they carry the
/// real shape of a design input: a target chain with coordinates plus an
/// **expandable segment** — a placeholder row whose `comp_id` is a range like
/// `50-120` and whose coordinates are `nan`, meaning "design between 50 and 120
/// residues here". A reader that rejects or zeroes those rows silently discards the
/// design specification, so that case is asserted directly.
@Suite("mmCIF reader")
struct MMCIFTests {

    struct Reference: Decodable, Sendable {
        struct Entry: Decodable, Sendable {
            let nAtoms: Int
            let columns: [String: [String]]
            enum CodingKeys: String, CodingKey {
                case nAtoms = "n_atoms"
                case columns
            }
        }
        let files: [String: Entry]
    }

    nonisolated(unsafe) static let reference: Reference? = {
        guard let url = Bundle.module.url(forResource: "cif_reference", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Reference.self, from: data)
    }()

    static func structureURL(_ name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "cif", subdirectory: "structures")
            ?? Bundle.module.url(forResource: name, withExtension: "cif")
    }

    static let cases = [
        ("protein_binder", "protein_binder_input"),
        ("enzyme_design", "enzyme_design_input"),
    ]

    @Test("atom_site columns match biotite", arguments: cases)
    func columnsMatchBiotite(refName: String, fileName: String) throws {
        let ref = try #require(Self.reference, "cif_reference.json missing")
        let entry = try #require(ref.files[refName], "no reference for \(refName)")
        let url = try #require(Self.structureURL(fileName), "missing \(fileName).cif")

        let file = try MMCIF.parse(contentsOf: url)
        for (column, want) in entry.columns {
            let got = try #require(file.column("atom_site", column),
                                   "\(refName): no column \(column)")
            #expect(got.count == entry.nAtoms, "\(refName).\(column): \(got.count) rows")
            guard got.count == want.count else { continue }
            // Numeric columns are compared as values; text columns verbatim.
            if column.hasPrefix("Cartn_") {
                for i in got.indices {
                    let a = Float(got[i]) ?? .nan, b = Float(want[i]) ?? .nan
                    let same = (a.isNaN && b.isNaN) || abs(a - b) < 1e-4
                    #expect(same, "\(refName).\(column)[\(i)]: \(got[i]) != \(want[i])")
                    if !same { break }
                }
            } else {
                let firstBad = got.indices.first { got[$0] != want[$0] }
                let detail = firstBad.map {
                    "\(refName).\(column)[\($0)]: '\(got[$0])' != '\(want[$0])'"
                } ?? ""
                #expect(firstBad == nil, "\(detail)")
            }
        }
    }

    @Test("an AtomArray is built from the binder-design input")
    func atomArrayFromBinderInput() throws {
        let url = try #require(Self.structureURL("protein_binder_input"))
        let array = try AtomArray.fromMMCIF(contentsOf: url)

        #expect(array.count == 925)
        #expect(array.chainID.map { Set($0) } == Set(["A", "B"]))

        // Chain A is the target: real residues, real coordinates.
        let chain = try #require(array.chainID)
        let targetIdx = (0 ..< array.count).filter { chain[$0] == "A" }
        #expect(targetIdx.count == 924)
        #expect(targetIdx.allSatisfy { array.coord[$0].x.isFinite })
        #expect(Set(targetIdx.map { array.resName[$0] }).isSubset(of: Vocabulary.standardAA),
                "the target should be canonical residues only")

        // Chain B is the expandable segment: one placeholder row, no coordinates,
        // and a comp_id that is a length range rather than a residue name.
        let segIdx = (0 ..< array.count).filter { chain[$0] == "B" }
        #expect(segIdx.count == 1)
        let seg = try #require(segIdx.first)
        #expect(array.resName[seg] == "50-120")
        #expect(array.coord[seg].x.isNaN, "an expandable segment carries no position")
        #expect(Vocabulary.unifiedAtom37.index(of: array.resName[seg]) == nil,
                "a length range is not a vocabulary residue")
    }

    // MARK: lexing

    @Test("a trailing apostrophe stays part of an atom name")
    func apostropheInAtomName() throws {
        // The case that breaks a naive split: nucleotide atom names end in '.
        let tokens = try MMCIF.tokenize("ATOM C C1' . A 1", line: 1)
        #expect(tokens == ["ATOM", "C", "C1'", ".", "A", "1"])
    }

    @Test("quotes delimit only when they open a token and close before whitespace")
    func quotingRules() throws {
        #expect(try MMCIF.tokenize("'a b' c", line: 1) == ["a b", "c"])
        #expect(try MMCIF.tokenize("\"x y\" z", line: 1) == ["x y", "z"])
        // An interior quote is literal, not a delimiter.
        #expect(try MMCIF.tokenize("O5' N1'", line: 1) == ["O5'", "N1'"])
        #expect(throws: MMCIFError.self) { try MMCIF.tokenize("'unterminated", line: 9) }
    }

    @Test("single-value categories are read")
    func singleValueCategories() throws {
        let url = try #require(Self.structureURL("enzyme_design_input"))
        let file = try MMCIF.parse(contentsOf: url)
        #expect(file.value("entry", "id") != nil)
        #expect(!file.dataBlock.isEmpty)
    }

    @Test("RFD4's design annotations survive the parse")
    func designAnnotations() throws {
        let url = try #require(Self.structureURL("protein_binder_input"))
        let file = try MMCIF.parse(contentsOf: url)
        // These are RFD4 extensions to atom_site, not standard mmCIF, and they carry
        // the hotspots and masks the featurizer will need.
        for column in ["mask_hotspot_1_atom", "mask_sequence_1_residue",
                       "mask_coordinate_1_atom", "annotation_expsegmin_1_residue",
                       "annotation_expsegmax_1_residue"] {
            let c = try #require(file.column("atom_site", column), "missing \(column)")
            #expect(c.count == 925)
        }
        // The expandable segment declares its own length bounds.
        let lo = try #require(file.column("atom_site", "mask_expsegmin_1_residue"))
        let hi = try #require(file.column("atom_site", "mask_expsegmax_1_residue"))
        #expect(lo.contains("True") || hi.contains("True"),
                "the design row should be flagged in the expandable-segment masks")
    }
}
