import Foundation

/// Compression-independent structure discovery and filenames.
///
/// Ported from `utils/structure_files.py`.
public enum StructureFiles {
    /// Recognised structure suffixes, longest-first.
    ///
    /// Transcribed from upstream's `STRUCTURE_FILE_SUFFIXES`, which is
    /// `atomworks.io.utils.io_utils.CIF_LIKE_EXTENSIONS` minus anything starting with
    /// `.json` ("in this codebase, jsons are never structure files, and instead may be
    /// ConditionSpec inputs") and minus `.pdb*`. Verified against the installed
    /// upstream package — note `.mmjson` survives the `.json` filter because it does
    /// not *start* with `.json`, and that every format carries a `.gzip` spelling
    /// alongside `.gz`.
    ///
    /// No entry is a suffix of another (`x.bcif.gz` does not end in `.cif.gz`), so
    /// match order affects only clarity, not correctness.
    public static let structureSuffixes = [
        ".mmjson.gzip", ".mmjson.zst", ".bcif.gzip", ".mmjson.gz",
        ".bcif.zst", ".cif.gzip", ".bcif.gz", ".cif.zst",
        ".mmjson", ".cif.gz", ".bcif", ".cif",
    ]

    /// Additional suffixes accepted only when `includePDB` is set.
    public static let pdbSuffixes = [".pdb.gzip", ".pdb.zst", ".pdb.gz", ".pdb"]

    static func allSuffixes(includePDB: Bool) -> [String] {
        includePDB ? structureSuffixes + pdbSuffixes : structureSuffixes
    }

    public enum StructureFileError: Error, Equatable, CustomStringConvertible {
        case unsupportedFilename(String)
        case ambiguous(key: String, first: String, second: String)

        public var description: String {
            switch self {
            case .unsupportedFilename(let p):
                "Unsupported structure filename: \(p)"
            case .ambiguous(let key, let first, let second):
                "Ambiguous structure files for \(key): \(first) and \(second). Keep one encoding."
            }
        }
    }

    /// Recognise exact structure suffixes, excluding backup and temporary files.
    ///
    /// Matching is on the *whole* suffix, so `design.cif.bak` and `design.cif.zst.tmp`
    /// are both rejected — the point of the function.
    public static func isStructureFile(_ path: String, includePDB: Bool = false) -> Bool {
        let lower = path.lowercased()
        // `structure_file_stem` always strips a cif-like suffix, pdb included, so a
        // `.pdb` path is a *structure filename* even when it is not an accepted input.
        return allSuffixes(includePDB: includePDB).contains { lower.hasSuffix($0) }
    }

    /// Remove the format and compression suffix, preserving dots in identifiers.
    ///
    /// `1abc.def_generated.cif.zst` → `1abc.def_generated`.
    public static func stem(_ path: String) throws -> String {
        let name = (path as NSString).lastPathComponent
        let lower = name.lowercased()
        // Upstream iterates the full cif-like list, pdb included, regardless of
        // include_pdb -- stems are computed for pdb paths too.
        for suffix in allSuffixes(includePDB: true) where lower.hasSuffix(suffix) {
            return String(name.dropLast(suffix.count))
        }
        throw StructureFileError.unsupportedFilename(path)
    }

    /// Sort structures and reject two encodings of the same sibling design.
    public static func unique(_ paths: [String]) throws -> [String] {
        let sorted = paths.sorted()
        var identities = [String: String]()
        for path in sorted {
            let url = URL(fileURLWithPath: path).standardizedFileURL
            let key = url.deletingLastPathComponent()
                .appendingPathComponent(try stem(path)).path
            if let first = identities[key] {
                throw StructureFileError.ambiguous(key: key, first: first, second: path)
            }
            identities[key] = path
        }
        return sorted
    }

    /// Find one sibling encoding of a nominal structure path, or `nil`.
    ///
    /// Returns `nil` — rather than throwing — when the parent directory does not
    /// exist, which is how callers probe for optional inputs.
    public static func find(_ nominalPath: String) throws -> String? {
        let url = URL(fileURLWithPath: nominalPath)
        let parent = url.deletingLastPathComponent()
        let wanted = try stem(nominalPath)

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: parent.path, isDirectory: &isDirectory),
              isDirectory.boolValue else { return nil }

        let entries = try FileManager.default.contentsOfDirectory(atPath: parent.path)
        let matches = entries
            .map { parent.appendingPathComponent($0).path }
            .filter { candidate in
                var isDir: ObjCBool = false
                let exists = FileManager.default.fileExists(atPath: candidate, isDirectory: &isDir)
                guard exists, !isDir.boolValue,
                      isStructureFile(candidate, includePDB: true) else { return false }
                return (try? stem(candidate)) == wanted
            }
        return try unique(matches).first
    }
}
