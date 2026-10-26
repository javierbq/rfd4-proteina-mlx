import Foundation
import Testing
@testable import RFD4Core

/// Ported from upstream `tests/utils/test_structure_files.py`.
///
/// Upstream docstring: "Compression-independent structure filenames and ambiguity checks."
@Suite("structure files")
struct StructureFilesTests {

    // MARK: test_structure_file_stem_preserves_dots

    @Test("the stem preserves dots in identifiers",
          arguments: [".cif", ".cif.gz", ".cif.zst", ".bcif", ".bcif.gz", ".bcif.zst"])
    func stemPreservesDots(suffix: String) throws {
        #expect(try StructureFiles.stem("1abc.def_generated\(suffix)") == "1abc.def_generated")
    }

    /// Not upstream. The suffix list was first reconstructed by inference and was
    /// WRONG — it missed the whole `.mmjson` family and every `.gzip` spelling. These
    /// come from the installed upstream `STRUCTURE_FILE_SUFFIXES` / `_ALL_SUFFIXES`,
    /// so the list cannot silently drift back.
    @Test("the full upstream suffix set is recognised")
    func fullUpstreamSuffixSet() throws {
        let nonPDB = [".mmjson", ".mmjson.gz", ".mmjson.gzip", ".mmjson.zst",
                      ".bcif", ".bcif.gz", ".bcif.gzip", ".bcif.zst",
                      ".cif", ".cif.gz", ".cif.gzip", ".cif.zst"]
        #expect(Set(StructureFiles.structureSuffixes) == Set(nonPDB))
        #expect(Set(StructureFiles.pdbSuffixes)
                == Set([".pdb", ".pdb.gz", ".pdb.gzip", ".pdb.zst"]))

        for suffix in nonPDB {
            #expect(StructureFiles.isStructureFile("design\(suffix)"), "\(suffix)")
            #expect(try StructureFiles.stem("design\(suffix)") == "design", "\(suffix)")
        }
        // `.mmjson` is kept although `.json` is filtered out: the filter is on names
        // that START with `.json`, and `.mmjson` does not.
        #expect(StructureFiles.isStructureFile("design.mmjson"))
        #expect(!StructureFiles.isStructureFile("design.json"))
    }

    // MARK: test_structure_file_stem_rejects_unsupported_suffix

    @Test("unsupported suffixes are rejected",
          arguments: ["design.txt", "design.cif.bak", "design.cif.zst.tmp",
                      "design.json", "design.json.gz", "design.json.gzip", "design.json.zst"])
    func rejectsUnsupportedSuffix(filename: String) {
        #expect(!StructureFiles.isStructureFile(filename))
        #expect(!StructureFiles.isStructureFile(filename, includePDB: true))
        #expect(throws: StructureFiles.StructureFileError.unsupportedFilename(filename)) {
            try StructureFiles.stem(filename)
        }
    }

    // MARK: test_unique_structure_files_rejects_sibling_encodings

    @Test("two encodings of one design are rejected", arguments: [".cif.gz", ".cif.zst"])
    func rejectsSiblingEncodings(suffix: String) throws {
        let dir = try TempDir()
        defer { dir.remove() }
        let paths = [dir.path("design.cif"), dir.path("design\(suffix)")]

        var thrown: Error?
        #expect(throws: StructureFiles.StructureFileError.self) {
            do { _ = try StructureFiles.unique(paths) } catch { thrown = error; throw error }
        }
        // Upstream asserts both offending paths appear in the message.
        let message = String(describing: try #require(thrown))
        for path in paths {
            #expect(message.contains(path), "message should name \(path)")
        }
        #expect(message.contains("Keep one encoding"))
    }

    // MARK: test_find_structure_file_missing_parent

    @Test("a missing parent directory yields nil, not an error")
    func findWithMissingParent() throws {
        let dir = try TempDir()
        defer { dir.remove() }
        #expect(try StructureFiles.find(dir.path("missing/design.cif")) == nil)
    }

    // MARK: not upstream — the positive paths the upstream file never exercises

    @Test("a sibling encoding is found regardless of compression")
    func findsSiblingEncoding() throws {
        let dir = try TempDir()
        defer { dir.remove() }
        let actual = dir.path("design.cif.zst")
        FileManager.default.createFile(atPath: actual, contents: Data())

        // Asked for the plain .cif; the compressed sibling is what exists.
        #expect(try StructureFiles.find(dir.path("design.cif")) == actual)
    }

    @Test("pdb is recognised only when asked for")
    func pdbGating() {
        #expect(!StructureFiles.isStructureFile("design.pdb"))
        #expect(StructureFiles.isStructureFile("design.pdb", includePDB: true))
        // The stem is still computable, because upstream strips pdb suffixes
        // regardless of the include_pdb gate.
        #expect(try! StructureFiles.stem("design.pdb") == "design")
    }

    @Test("unique sorts and keeps distinct designs")
    func uniqueSortsDistinctDesigns() throws {
        let dir = try TempDir()
        defer { dir.remove() }
        let paths = [dir.path("b.cif"), dir.path("a.cif.zst")]
        #expect(try StructureFiles.unique(paths) == paths.sorted())
    }
}

/// A unique temporary directory, removed on `remove()`.
struct TempDir {
    let url: URL

    init() throws {
        url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rfd4-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    func path(_ component: String) -> String {
        url.appendingPathComponent(component).path
    }

    func remove() { try? FileManager.default.removeItem(at: url) }
}
