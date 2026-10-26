import Foundation

/// A minimal mmCIF reader, sufficient for RFD4's inputs and outputs.
///
/// This is not a general CIF library. It reads the categories this port needs —
/// chiefly `atom_site` — and it follows CIF's quoting rules closely enough that
/// atom names like `C1'` and `O5'` survive, which a naive whitespace split destroys:
/// a quote delimits a value only when it opens a token and its partner is followed
/// by whitespace. Nucleotide atom names are full of apostrophes, so getting this
/// wrong corrupts every nucleic-acid structure while leaving proteins fine.
///
/// RFD4's own files extend `atom_site` with design annotations — hotspots, sequence
/// and coordinate masks, and expandable-segment bounds — so those columns are read
/// through the same generic path rather than special-cased.
public struct MMCIFFile {
    /// Category name (without the leading `_`) → column name → values, one per row.
    /// A non-loop category is stored as a single row.
    public let categories: [String: [String: [String]]]
    public let dataBlock: String

    public func column(_ category: String, _ name: String) -> [String]? {
        categories[category]?[name]
    }

    /// The first value of a single-row category, e.g. `entry.id`.
    public func value(_ category: String, _ name: String) -> String? {
        column(category, name)?.first
    }

    public var rowCount: [String: Int] {
        categories.mapValues { $0.values.first?.count ?? 0 }
    }
}

public enum MMCIFError: Error, CustomStringConvertible {
    case unterminatedQuote(line: Int)
    case loopWithoutColumns(line: Int)
    case raggedLoop(category: String, expected: Int, got: Int)
    case missingCategory(String)
    case missingColumn(category: String, column: String)

    public var description: String {
        switch self {
        case .unterminatedQuote(let l): "unterminated quoted value on line \(l)"
        case .loopWithoutColumns(let l): "loop_ with no column names, line \(l)"
        case .raggedLoop(let c, let e, let g):
            "loop \(c) has \(g) values, not a multiple of its \(e) columns"
        case .missingCategory(let c): "mmCIF has no category _\(c)"
        case .missingColumn(let c, let n): "category _\(c) has no column \(n)"
        }
    }
}

public enum MMCIF {
    /// CIF nulls. Both mean "no value"; this reader does not distinguish them.
    public static let nullTokens: Set<String> = [".", "?"]

    public static func parse(contentsOf url: URL) throws -> MMCIFFile {
        try parse(String(contentsOf: url, encoding: .utf8))
    }

    public static func parse(_ text: String) throws -> MMCIFFile {
        var categories = [String: [String: [String]]]()
        var dataBlock = ""

        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        var i = 0

        while i < lines.count {
            let raw = lines[i]
            let line = raw.trimmingCharacters(in: .whitespaces)
            i += 1

            if line.isEmpty || line.hasPrefix("#") { continue }

            if line.lowercased().hasPrefix("data_") {
                dataBlock = String(line.dropFirst(5))
                continue
            }

            if line.lowercased() == "loop_" {
                // Column names first, then rows until the next directive.
                var columns: [(category: String, name: String)] = []
                while i < lines.count {
                    let peek = lines[i].trimmingCharacters(in: .whitespaces)
                    guard peek.hasPrefix("_") else { break }
                    columns.append(splitTag(peek))
                    i += 1
                }
                guard !columns.isEmpty else { throw MMCIFError.loopWithoutColumns(line: i) }

                var values: [String] = []
                while i < lines.count {
                    let peek = lines[i].trimmingCharacters(in: .whitespaces)
                    if peek.hasPrefix("_") || peek.lowercased() == "loop_"
                        || peek.lowercased().hasPrefix("data_") { break }
                    if peek == "#" { i += 1; break }
                    if peek.isEmpty { i += 1; continue }
                    if peek.hasPrefix(";") {
                        values.append(try readSemicolonText(lines, &i))
                        continue
                    }
                    values.append(contentsOf: try tokenize(peek, line: i))
                    i += 1
                }

                let category = columns[0].category
                guard values.count % columns.count == 0 else {
                    throw MMCIFError.raggedLoop(category: category,
                                                expected: columns.count, got: values.count)
                }
                var table = categories[category] ?? [:]
                let rows = values.count / columns.count
                for (c, col) in columns.enumerated() {
                    var out = table[col.name] ?? []
                    out.reserveCapacity(rows)
                    for r in 0 ..< rows { out.append(values[r * columns.count + c]) }
                    table[col.name] = out
                }
                categories[category] = table
                continue
            }

            if line.hasPrefix("_") {
                // `_category.item value`, with the value possibly on the next line.
                let parts = try tokenize(line, line: i)
                let tag = splitTag(parts[0])
                var value: String
                if parts.count > 1 {
                    value = parts[1 ..< parts.count].joined(separator: " ")
                } else if i < lines.count,
                          lines[i].trimmingCharacters(in: .whitespaces).hasPrefix(";") {
                    value = try readSemicolonText(lines, &i)
                } else if i < lines.count {
                    value = try tokenize(
                        lines[i].trimmingCharacters(in: .whitespaces), line: i).first ?? ""
                    i += 1
                } else {
                    value = ""
                }
                categories[tag.category, default: [:]][tag.name] = [value]
                continue
            }
        }

        return MMCIFFile(categories: categories, dataBlock: dataBlock)
    }

    // MARK: - lexing

    private static func splitTag(_ token: String) -> (category: String, name: String) {
        let body = token.hasPrefix("_") ? String(token.dropFirst()) : token
        guard let dot = body.firstIndex(of: ".") else { return (body, "") }
        return (String(body[body.startIndex ..< dot]), String(body[body.index(after: dot)...]))
    }

    /// A `;`-delimited multi-line value.
    private static func readSemicolonText(
        _ lines: [Substring], _ i: inout Int
    ) throws -> String {
        var body: [String] = []
        var first = String(lines[i])
        if first.hasPrefix(";") { first.removeFirst() }
        body.append(first)
        i += 1
        while i < lines.count {
            let l = String(lines[i])
            i += 1
            if l.trimmingCharacters(in: .whitespaces).hasPrefix(";") { break }
            body.append(l)
        }
        return body.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Split one line into CIF values.
    ///
    /// A `'` or `"` opens a quoted value only at the start of a token, and closes it
    /// only when followed by whitespace or end of line. That is what keeps `C1'` and
    /// `O5'` intact — the trailing apostrophe is part of the name, not a delimiter.
    static func tokenize(_ line: String, line number: Int) throws -> [String] {
        var out: [String] = []
        let chars = Array(line)
        var i = 0

        while i < chars.count {
            while i < chars.count, chars[i] == " " || chars[i] == "\t" { i += 1 }
            guard i < chars.count else { break }

            if chars[i] == "#" { break }   // trailing comment

            if chars[i] == "'" || chars[i] == "\"" {
                let quote = chars[i]
                var j = i + 1
                var closed = false
                while j < chars.count {
                    if chars[j] == quote,
                       j + 1 >= chars.count || chars[j + 1] == " " || chars[j + 1] == "\t" {
                        closed = true
                        break
                    }
                    j += 1
                }
                guard closed else { throw MMCIFError.unterminatedQuote(line: number) }
                out.append(String(chars[(i + 1) ..< j]))
                i = j + 1
                continue
            }

            var j = i
            while j < chars.count, chars[j] != " ", chars[j] != "\t" { j += 1 }
            out.append(String(chars[i ..< j]))
            i = j
        }
        return out
    }
}

// MARK: - atom_site -> AtomArray

extension AtomArray {
    /// Build an `AtomArray` from an mmCIF's `atom_site` loop.
    ///
    /// Uses the `label_*` identifiers, which is what atomworks' parser keys on.
    /// Coordinates may be `nan`: RFD4 writes expandable segments — "design 50 to 120
    /// residues here" — as placeholder rows with no position, so a reader that
    /// rejects or zeroes them loses the design spec entirely.
    public static func fromMMCIF(_ file: MMCIFFile) throws -> AtomArray {
        func col(_ name: String) throws -> [String] {
            guard let c = file.column("atom_site", name) else {
                guard file.categories["atom_site"] != nil else {
                    throw MMCIFError.missingCategory("atom_site")
                }
                throw MMCIFError.missingColumn(category: "atom_site", column: name)
            }
            return c
        }

        let resName = try col("label_comp_id")
        let atomName = try col("label_atom_id")
        let xs = try col("Cartn_x"), ys = try col("Cartn_y"), zs = try col("Cartn_z")

        func f(_ s: String) -> Float {
            MMCIF.nullTokens.contains(s) ? .nan : (Float(s) ?? .nan)
        }
        let coords = (0 ..< resName.count).map { Vec3(f(xs[$0]), f(ys[$0]), f(zs[$0])) }

        var array = AtomArray(resName: resName, atomName: atomName, coord: coords)
        array.element = file.column("atom_site", "type_symbol")
        array.chainID = file.column("atom_site", "label_asym_id")
        array.resID = file.column("atom_site", "label_seq_id")?.map { Int($0) ?? -1 }
        array.occupancy = file.column("atom_site", "occupancy")?.map(f)
        array.bFactor = file.column("atom_site", "B_iso_or_equiv")?.map(f)
        return array
    }

    public static func fromMMCIF(contentsOf url: URL) throws -> AtomArray {
        try fromMMCIF(MMCIF.parse(contentsOf: url))
    }
}
