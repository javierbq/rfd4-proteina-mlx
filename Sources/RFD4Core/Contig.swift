import Foundation

/// The contig string: one comma-separated spec that both subsets an input
/// structure and inserts the segments to be designed.
///
/// `"50-120, /0, A17-131"` is the shipped protein-binder recipe — design a new
/// chain of 50 to 120 residues, then a chain break, then residues 17 to 131 of the
/// input's chain A kept verbatim. This is the only place a design request becomes a
/// token layout, so the grammar is ported exactly rather than approximated:
/// `contig_spec.py` classifies each token **by shape**, and the shapes overlap in a
/// way that punishes a looser reader. `A1-10` is a preserved range only because it
/// starts with a letter; `50-120` is a length range only because it does not;
/// `X1:5-8` is a *named* expandable segment, where the name is everything before
/// the colon and may itself start with a digit.
public enum Contig {

    /// The chain type a `P` / `R` / `D` suffix selects for an expandable segment.
    public enum SegmentChainType: String, Sendable, Equatable {
        case polypeptide = "P", rna = "R", dna = "D"
    }

    /// A range of the input structure, kept verbatim.
    public struct PreservedSegment: Equatable, Sendable {
        public let chainID: String
        public let start: Int
        public let end: Int
        public var length: Int { end - start + 1 }

        public init(chainID: String, start: Int, end: Int) {
            self.chainID = chainID; self.start = start; self.end = end
        }
    }

    /// A stretch to be designed, whose length is drawn from `min...max`.
    public struct ExpandableSegment: Equatable, Sendable {
        public let name: String?
        public let min: Int
        public let max: Int
        /// An explicit suffix; `nil` means "take the run's default".
        public let chainType: SegmentChainType?

        public init(name: String?, min: Int, max: Int, chainType: SegmentChainType? = nil) {
            self.name = name; self.min = min; self.max = max; self.chainType = chainType
        }

        public var isFixedLength: Bool { min == max }
    }

    public enum Token: Equatable, Sendable {
        case chainBreak
        case preserved(PreservedSegment)
        case expandable(ExpandableSegment)
    }

    /// One output chain: the tokens between two chain breaks, plus the id they land on.
    public struct ChainRun: Equatable, Sendable {
        public let tokens: [Token]
        public let chainID: String

        public init(tokens: [Token], chainID: String) {
            self.tokens = tokens; self.chainID = chainID
        }
    }

    // MARK: - parsing

    private static let lengthRangeChars = Set("0123456789-")

    /// `atomworks.constants.MASKED`, reserved to mark an already-realized sentinel.
    static let maskedSentinel = "<M>"

    /// Classify and parse one token, by shape.
    public static func parseToken(_ raw: String) throws -> Token {
        // `.whitespaces` excludes newlines; Python's `str.strip()` does not. A contig
        // written across lines in a YAML block arrives with them attached.
        let token = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else { throw ContigError.emptyToken }
        if token == "/0" { return .chainBreak }

        // A colon means a NAMED expandable segment, and that test comes first: the
        // name may begin with a letter (`X1:5-8`) or a digit, so checking the first
        // character before the colon would misfile both ways.
        if let colon = token.firstIndex(of: ":") {
            let name = String(token[token.startIndex ..< colon])
            guard !name.isEmpty else { throw ContigError.malformedExpandable(token) }
            // `atomworks.constants.MASKED` is the literal "<M>", not the word
            // MASKED. Checking for the word gets it exactly backwards: it admits
            // the sentinel upstream rejects and rejects a name upstream allows.
            guard name != Self.maskedSentinel else {
                throw ContigError.reservedSegmentName(token)
            }
            let body = String(token[token.index(after: colon)...])
            let (lo, hi, type) = try parseLengthAndSuffix(body, token)
            return .expandable(ExpandableSegment(name: name, min: lo, max: hi, chainType: type))
        }
        if token.first!.isLetter {
            return .preserved(try parsePreserved(token))
        }
        let (lo, hi, type) = try parseLengthAndSuffix(token, token)
        return .expandable(ExpandableSegment(name: nil, min: lo, max: hi, chainType: type))
    }

    private static func parseLengthAndSuffix(
        _ raw: String, _ token: String
    ) throws -> (Int, Int, SegmentChainType?) {
        var body = raw
        var chainType: SegmentChainType?
        if let last = body.last, let t = SegmentChainType(rawValue: String(last)) {
            chainType = t
            body.removeLast()
        }
        guard !body.isEmpty, Set(body).isSubset(of: lengthRangeChars) else {
            throw ContigError.malformedExpandable(token)
        }
        let parts = body.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
        let lo: Int, hi: Int
        switch parts.count {
        case 1:
            guard let v = Int(parts[0]) else { throw ContigError.malformedExpandable(token) }
            lo = v; hi = v
        case 2:
            guard let a = Int(parts[0]), let b = Int(parts[1]) else {
                throw ContigError.malformedExpandable(token)
            }
            lo = a; hi = b
        default:
            throw ContigError.malformedExpandable(token)
        }
        guard hi >= lo else { throw ContigError.reversedRange(token, lo, hi) }
        return (lo, hi, chainType)
    }

    private static func parsePreserved(_ token: String) throws -> PreservedSegment {
        // The chain id is the LEADING run of letters, so a multi-character id such as
        // `AB12-20` splits correctly where "first character" would not.
        let chars = Array(token)
        var i = 0
        while i < chars.count, chars[i].isLetter { i += 1 }
        guard i > 0 else { throw ContigError.malformedPreserved(token) }
        let chainID = String(chars[0 ..< i])
        let rest = String(chars[i...])
        let parts = rest.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
        let start: Int, end: Int
        switch parts.count {
        case 1:
            guard parts[0].allSatisfy(\.isNumber), let v = Int(parts[0]) else {
                throw ContigError.malformedPreserved(token)
            }
            start = v; end = v
        case 2:
            guard parts[0].allSatisfy(\.isNumber), parts[1].allSatisfy(\.isNumber),
                  let a = Int(parts[0]), let b = Int(parts[1]) else {
                throw ContigError.malformedPreserved(token)
            }
            start = a; end = b
        default:
            throw ContigError.malformedPreserved(token)
        }
        guard end >= start else { throw ContigError.reversedRange(token, start, end) }
        return PreservedSegment(chainID: chainID, start: start, end: end)
    }

    public static func parse(_ contig: String) throws -> [Token] {
        try contig.split(separator: ",", omittingEmptySubsequences: false)
            .map { try parseToken(String($0)) }
    }

    // MARK: - chain runs

    /// Split into output chains and give each one its id.
    ///
    /// Assignment is **chain-run-first**, not a left-to-right walk: a run's id comes
    /// from whichever preserved segment it contains, and that segment may appear
    /// after the expandable ones it names. In `"A1-10, /0, E1:1-20, B1-10"` the
    /// segment `E1` lands on chain B, which is only knowable once the whole run has
    /// been read.
    public static func chainRuns(_ contig: String) throws -> [ChainRun] {
        let tokens = try parse(contig)

        let names = tokens.compactMap { token -> String? in
            if case .expandable(let e) = token { return e.name }
            return nil
        }
        var counts: [String: Int] = [:]
        for n in names { counts[n, default: 0] += 1 }
        let dupes = counts.filter { $0.value > 1 }.keys.sorted()
        guard dupes.isEmpty else { throw ContigError.duplicateSegmentNames(dupes) }

        var runs: [[Token]] = [[]]
        for token in tokens {
            if token == .chainBreak { runs.append([]) } else { runs[runs.count - 1].append(token) }
        }
        runs = runs.filter { !$0.isEmpty }
        guard !runs.isEmpty else { throw ContigError.noChainRuns }

        try validateNoOverlappingPreserved(runs)

        var claimed = Set<String>()
        for run in runs {
            for token in run { if case .preserved(let p) = token { claimed.insert(p.chainID) } }
        }
        var synthetic = syntheticChainIDs(excluding: claimed).makeIterator()

        var seen = Set<String>()
        var out: [ChainRun] = []
        for run in runs {
            let ids = Set(run.compactMap { token -> String? in
                if case .preserved(let p) = token { return p.chainID }
                return nil
            })
            let chainID: String
            if ids.isEmpty {
                guard let fresh = synthetic.next() else { throw ContigError.chainIDsExhausted }
                chainID = fresh
            } else if ids.count > 1 {
                throw ContigError.mixedChains(ids.sorted())
            } else {
                chainID = ids.first!
            }
            guard !seen.contains(chainID) else { throw ContigError.duplicateOutputChain(chainID) }
            seen.insert(chainID)
            out.append(ChainRun(tokens: run, chainID: chainID))
        }
        return out
    }

    /// Two preserved tokens may not claim the same input residue twice.
    private static func validateNoOverlappingPreserved(_ runs: [[Token]]) throws {
        var byChain: [String: [PreservedSegment]] = [:]
        for run in runs {
            for token in run {
                if case .preserved(let p) = token { byChain[p.chainID, default: []].append(p) }
            }
        }
        for (chain, segments) in byChain.sorted(by: { $0.key < $1.key }) {
            let sorted = segments.sorted { $0.start < $1.start }
            for (a, b) in zip(sorted, sorted.dropFirst()) where b.start <= a.end {
                throw ContigError.overlappingPreserved(chain, a, b)
            }
        }
    }

    /// `A`, `B`, … `Z`, `AA`, `AB`, … skipping anything the contig already claims.
    static func syntheticChainIDs(excluding claimed: Set<String>) -> AnySequence<String> {
        AnySequence { () -> AnyIterator<String> in
            let alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            var width = 1
            var counter = 0
            return AnyIterator {
                while width <= 4 {
                    let limit = Int(pow(26.0, Double(width)))
                    while counter < limit {
                        var n = counter
                        var chars: [Character] = []
                        for _ in 0 ..< width {
                            chars.append(alphabet[n % 26]); n /= 26
                        }
                        counter += 1
                        let id = String(chars.reversed())
                        if !claimed.contains(id) { return id }
                    }
                    width += 1
                    counter = 0
                }
                return nil
            }
        }
    }

    // MARK: - realization

    /// One segment of the realized layout, with a concrete length.
    public struct RealizedSegment: Equatable, Sendable {
        public let chainID: String
        /// Which output chain this belongs to, counting from 0 in contig order.
        public let chainIndex: Int
        /// The input residue numbers this segment preserves, ascending.
        /// Empty for a designed segment.
        public let residues: [Int]
        /// How many tokens this segment contributes.
        public let length: Int
        /// `nil` for a designed segment; the requested input range for a preserved one.
        public let preserved: PreservedSegment?
        /// An explicit `P`/`R`/`D` suffix on a designed segment, if it carried one.
        public let chainType: SegmentChainType?
        public var isDesigned: Bool { preserved == nil }
    }

    /// The token layout for one design: every segment with its length fixed.
    ///
    /// An expandable segment's length is **drawn**, so two runs of the same contig are
    /// two different problems. The draw is explicit here rather than implicit in a
    /// global generator, so a design can be reproduced.
    ///
    /// A preserved segment's length is **not** `end - start + 1`. Upstream selects
    /// `chain_id == c and start <= res_id <= end` from the real structure and counts
    /// the residues that are actually there, so an unmodelled loop — which is the norm
    /// in a crystal structure, not the exception — makes the segment shorter than its
    /// range. Computing the length arithmetically instead inserts ghost tokens that
    /// carry no atoms: an all-zero atom37 mask, NaN coordinates and NaN sequence
    /// probabilities, shifting the chain break and every conditioning array after
    /// them, with nothing to indicate it happened.
    ///
    /// - Parameter residuesInChain: the residue numbers present in a given input
    ///   chain, in any order. This is the structure talking; there is deliberately no
    ///   overload that guesses, because guessing is the defect.
    public static func realize(
        _ contig: String,
        residuesInChain: (String) -> [Int],
        draw: (Int, Int) -> Int
    ) throws -> [RealizedSegment] {
        var out: [RealizedSegment] = []
        for (index, run) in try chainRuns(contig).enumerated() {
            for token in run.tokens {
                switch token {
                case .chainBreak:
                    continue

                case .preserved(let p):
                    let available = residuesInChain(p.chainID)
                    guard let lo = available.min(), let hi = available.max() else {
                        throw ContigError.segmentMatchedNothing(p)
                    }
                    // Upstream raises for a range running off either end of the
                    // chain, calling it a typo rather than silently clipping.
                    guard p.start >= lo, p.end <= hi else {
                        throw ContigError.segmentOutsideChain(p, chainRange: lo ... hi)
                    }
                    let kept = available.filter { $0 >= p.start && $0 <= p.end }.sorted()
                    guard !kept.isEmpty else { throw ContigError.segmentMatchedNothing(p) }
                    out.append(RealizedSegment(
                        chainID: run.chainID, chainIndex: index,
                        residues: kept, length: kept.count, preserved: p,
                        chainType: nil))

                case .expandable(let e):
                    let n = e.isFixedLength ? e.min : draw(e.min, e.max)
                    guard n >= e.min, n <= e.max else {
                        throw ContigError.drawOutOfRange(e, drawn: n)
                    }
                    out.append(RealizedSegment(
                        chainID: run.chainID, chainIndex: index,
                        residues: [], length: n, preserved: nil,
                        chainType: e.chainType))
                }
            }
        }
        return out
    }
}

public enum ContigError: Error, Equatable, CustomStringConvertible {
    case emptyToken
    case malformedPreserved(String)
    case malformedExpandable(String)
    case reservedSegmentName(String)
    case reversedRange(String, Int, Int)
    case duplicateSegmentNames([String])
    case noChainRuns
    case mixedChains([String])
    case duplicateOutputChain(String)
    case overlappingPreserved(String, Contig.PreservedSegment, Contig.PreservedSegment)
    case chainIDsExhausted
    case segmentMatchedNothing(Contig.PreservedSegment)
    case segmentOutsideChain(Contig.PreservedSegment, chainRange: ClosedRange<Int>)
    case drawOutOfRange(Contig.ExpandableSegment, drawn: Int)

    public var description: String {
        switch self {
        case .emptyToken:
            return "Empty contig token (check for stray/trailing commas)."
        case .malformedPreserved(let t):
            return "Malformed preserved-segment token: \(t)"
        case .malformedExpandable(let t):
            return "Malformed expandable-segment token: \(t)"
        case .reservedSegmentName(let t):
            return "Expandable segment name cannot be MASKED: \(t)"
        case .reversedRange(let t, let a, let b):
            return "Segment \(t) has end < start (\(b) < \(a))."
        case .duplicateSegmentNames(let n):
            return "Duplicate expandable-segment name(s) in contig: \(n)"
        case .noChainRuns:
            return "Contig string produced no chain runs (only chainbreaks/empty tokens)."
        case .mixedChains(let ids):
            return """
                An output chain mixes preserved segments from multiple original chains \
                \(ids) with no `/0` between them — insert a `/0` to split them into \
                separate output chains.
                """
        case .duplicateOutputChain(let id):
            return "Contig produces two separate output chains both named \(id)."
        case .overlappingPreserved(let chain, let a, let b):
            return """
                Preserved segments \(chain)\(a.start)-\(a.end) and \(chain)\(b.start)-\(b.end) \
                overlap; each input residue may be used once.
                """
        case .chainIDsExhausted:
            return "Ran out of synthetic chain ids."
        case .segmentMatchedNothing(let p):
            return """
                Preserved segment '\(p.chainID)\(p.start)-\(p.end)' matched no atoms \
                in the input structure.
                """
        case .segmentOutsideChain(let p, let range):
            return """
                Preserved segment '\(p.chainID)\(p.start)-\(p.end)' runs outside chain \
                \(p.chainID), whose residues are \(range.lowerBound)-\(range.upperBound).
                """
        case .drawOutOfRange(let e, let n):
            return "Design-length draw returned \(n), outside \(e.min)...\(e.max)."
        }
    }
}
