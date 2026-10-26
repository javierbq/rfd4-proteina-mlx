import Testing
@testable import RFD4Core

/// Ported from upstream `tests/test_nucleic_ss_input.py`.
///
/// Upstream docstring: "Dot-bracket parsing across letter-bracket case conventions."
@Suite("dot-bracket parsing")
struct DotBracketTests {
    /// `nil` expected pairs means the input must be rejected.
    static let cases: [(dbn: String, expected: [(Int, Int)]?)] = [
        ("((..))", [(0, 5), (1, 4)]),            // nested
        ("(([[))]]", [(0, 5), (1, 4), (2, 7), (3, 6)]),  // crossing symbol families
        ("A..a", [(0, 3)]),                      // uppercase opens (Vienna/bpRNA)
        ("a..A", [(0, 3)]),                      // lowercase opens (Eterna/OpenKnot)
        ("(a.).A", [(0, 3), (1, 5)]),            // letter family crossing a symbol family
        ("((.)", nil),                           // unmatched opener
        ("(.))", nil),                           // unmatched closer
        ("a..a", nil),                           // same case twice cannot pair
        ("((.-.))", nil),                        // unrecognized character
    ]

    @Test("both case conventions parse; malformed input raises", arguments: cases)
    func parseDotBracket(dbn: String, expected: [(Int, Int)]?) throws {
        guard let expected else {
            #expect(throws: DotBracket.ParseError.self) { try DotBracket.parse(dbn) }
            return
        }

        let parsed = try DotBracket.parse(dbn)
        let sorted = parsed.pairs.sorted { $0 < $1 || ($0.0 == $1.0 && $0.1 < $1.1) }
        #expect(sorted.count == expected.count)
        #expect(zip(sorted, expected).allSatisfy { $0 == $1 })

        // Nothing silently dropped: every position is either paired or marked unpaired.
        let covered = Set(parsed.unpaired).union(parsed.pairs.flatMap { [$0.0, $0.1] })
        #expect(covered == Set(0..<dbn.count))
    }

    // MARK: not upstream — pins the error messages and the case-resolution rule

    @Test("the opener is whichever case appears first")
    func openerCaseIsResolvedPerString() {
        // Vienna/bpRNA: uppercase first, so uppercase opens.
        #expect(DotBracket.openerFor(Array("A..a"))["a"] == "A")
        // Eterna/OpenKnot: lowercase first, so lowercase opens.
        #expect(DotBracket.openerFor(Array("a..A"))["A"] == "a")
    }

    @Test("all four symbol families are supported")
    func symbolFamilies() throws {
        for (open, close) in DotBracket.symbolPairs {
            let parsed = try DotBracket.parse("\(open).\(close)")
            #expect(parsed.pairs.count == 1)
            #expect(parsed.pairs[0] == (0, 2))
            #expect(parsed.unpaired == [1])
        }
    }

    @Test("error messages match upstream wording")
    func errorMessages() {
        #expect(DotBracket.ParseError.unmatchedClosing(")", position: 3).description
                == "unmatched closing bracket ')' at position 3 in dot-bracket string")
        #expect(DotBracket.ParseError.unrecognizedCharacter("-", position: 3).description
                == "unrecognized dot-bracket character '-' at position 3")
    }

    @Test("an empty string parses to nothing")
    func emptyString() throws {
        let parsed = try DotBracket.parse("")
        #expect(parsed.pairs.isEmpty)
        #expect(parsed.unpaired.isEmpty)
    }
}
