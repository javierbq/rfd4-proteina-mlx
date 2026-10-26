import Foundation

/// Dot-bracket nucleic-acid secondary structure parsing.
///
/// Ported from `inference_utils/nucleic_ss_input.py` (`_opener_for`, `parse_dot_bracket`).
///
/// Supports `()`, `[]`, `{}`, `<>` plus letter families (`Aa`, `Bb`, …) for
/// higher-order pseudoknots, in **either** case convention: Vienna/bpRNA write
/// `A...a` (uppercase opens) while Eterna/OpenKnot write `a...A` (lowercase opens).
/// Both are well-formed, so the opener is whichever case appears first for that letter.
public enum DotBracket {
    static let symbolPairs: [Character: Character] = ["(": ")", "[": "]", "{": "}", "<": ">"]

    public struct Parsed: Equatable, Sendable {
        /// 0-based `(i, j)` index pairs, in the order closers were encountered.
        public let pairs: [(Int, Int)]
        /// 0-based indices of `.` characters.
        public let unpaired: [Int]

        public static func == (a: Parsed, b: Parsed) -> Bool {
            a.unpaired == b.unpaired
                && a.pairs.count == b.pairs.count
                && zip(a.pairs, b.pairs).allSatisfy { $0 == $1 }
        }
    }

    public enum ParseError: Error, Equatable, CustomStringConvertible {
        case unmatchedClosing(Character, position: Int)
        case unrecognizedCharacter(Character, position: Int)
        case unmatchedOpening(String)

        public var description: String {
            switch self {
            case .unmatchedClosing(let c, let i):
                "unmatched closing bracket '\(c)' at position \(i) in dot-bracket string"
            case .unrecognizedCharacter(let c, let i):
                "unrecognized dot-bracket character '\(c)' at position \(i)"
            case .unmatchedOpening(let detail):
                "unmatched opening bracket(s) in dot-bracket string: \(detail)"
            }
        }
    }

    /// Map each closing character to its opener, resolving letter case for this
    /// specific string.
    ///
    /// A letter whose partner case is missing leaves the string unbalanced; either
    /// assignment surfaces that as an unmatched-bracket error downstream, which is
    /// why no validation happens here.
    static func openerFor(_ dotBracket: [Character]) -> [Character: Character] {
        var openerFor = [Character: Character]()
        for (open, close) in symbolPairs { openerFor[close] = open }

        let letters = Set(dotBracket.filter(\.isLetter).map { Character($0.lowercased()) })
        for lower in letters {
            let upper = Character(lower.uppercased())
            let iLower = dotBracket.firstIndex(of: lower)
            let iUpper = dotBracket.firstIndex(of: upper)
            if let iUpper, iLower == nil || iUpper < iLower! {
                openerFor[lower] = upper
            } else {
                openerFor[upper] = lower
            }
        }
        return openerFor
    }

    /// Parse into base pairs and unpaired positions.
    ///
    /// Malformed conditioning is never dropped silently — an ignored bracket would
    /// quietly weaken the requested secondary structure — so this throws rather than
    /// skipping.
    public static func parse(_ dotBracket: String) throws -> Parsed {
        let chars = Array(dotBracket)
        let openerFor = openerFor(chars)
        let openers = Set(openerFor.values)

        var stack = [Character: [Int]]()
        var pairs = [(Int, Int)]()
        var unpaired = [Int]()

        for (i, ch) in chars.enumerated() {
            if ch == "." {
                unpaired.append(i)
            } else if openers.contains(ch) {
                stack[ch, default: []].append(i)
            } else if let open = openerFor[ch] {
                guard var slot = stack[open], !slot.isEmpty else {
                    throw ParseError.unmatchedClosing(ch, position: i)
                }
                let start = slot.removeLast()
                stack[open] = slot
                pairs.append((start, i))
            } else {
                throw ParseError.unrecognizedCharacter(ch, position: i)
            }
        }

        let dangling = stack.filter { !$0.value.isEmpty }
        guard dangling.isEmpty else {
            let detail = dangling.keys.sorted()
                .map { "'\($0)' at \(dangling[$0]!)" }
                .joined(separator: ", ")
            throw ParseError.unmatchedOpening(detail)
        }
        return Parsed(pairs: pairs, unpaired: unpaired)
    }
}
