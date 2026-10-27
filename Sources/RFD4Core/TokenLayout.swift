import Foundation

/// The token layout of one design problem: what each token is, and how the tokens
/// are connected.
///
/// This is the piece that stands between a design request and the model. Everything
/// here is derived from the realized contig rather than defaulted, because the
/// defaults are not neutral: a zero in `tokenBondPathLength` means "disconnected",
/// not "unknown", and filling the connectivity with zeros makes the network
/// generate a cloud of unbonded fragments. The de-novo path learned that the
/// expensive way; a two-chain binder has more ways to get it wrong, not fewer.
public struct TokenLayout: Equatable, Sendable {

    /// What one token is and where it came from.
    public struct Token: Equatable, Sendable {
        /// Which output chain, counting from 0 in contig order.
        public let chainIndex: Int
        public let chainID: String
        /// Which realized segment, counting from 0 across the whole layout.
        public let segmentIndex: Int
        /// The input residue number this token preserves; `nil` for a designed one.
        public let inputResidue: Int?
        public var isDesigned: Bool { inputResidue == nil }

        public init(chainIndex: Int, chainID: String, segmentIndex: Int, inputResidue: Int?) {
            self.chainIndex = chainIndex
            self.chainID = chainID
            self.segmentIndex = segmentIndex
            self.inputResidue = inputResidue
        }
    }

    public let tokens: [Token]
    /// Upstream's BFS cutoff is 64, so a connected pair beyond it is stored as 65.
    public let pathLengthCap: Int

    public var count: Int { tokens.count }
    public var chainCount: Int { (tokens.map(\.chainIndex).max() ?? -1) + 1 }

    public init(tokens: [Token], pathLengthCap: Int = 65) {
        self.tokens = tokens
        self.pathLengthCap = pathLengthCap
    }

    /// Expand a realized contig into one token per residue.
    ///
    /// A preserved token's `inputResidue` is read off the residues the segment
    /// actually resolved to, never computed as `start + offset`: a chain with an
    /// unmodelled loop numbers its residues with a hole in it, and the arithmetic
    /// form would point every token after the hole at the wrong residue.
    public init(_ segments: [Contig.RealizedSegment], pathLengthCap: Int = 65) {
        var tokens: [Token] = []
        for (segmentIndex, segment) in segments.enumerated() {
            for offset in 0 ..< segment.length {
                tokens.append(Token(
                    chainIndex: segment.chainIndex,
                    chainID: segment.chainID,
                    segmentIndex: segmentIndex,
                    inputResidue: segment.isDesigned ? nil : segment.residues[offset]))
            }
        }
        self.init(tokens: tokens, pathLengthCap: pathLengthCap)
    }

    /// Build from a contig string, resolving preserved segments against the structure.
    ///
    /// - Parameter residuesInChain: the residue numbers present in a given input
    ///   chain. See `Contig.realize` for why this is required rather than inferred.
    public static func make(
        contig: String,
        residuesInChain: (String) -> [Int],
        draw: (Int, Int) -> Int,
        pathLengthCap: Int = 65
    ) throws -> TokenLayout {
        let segments = try Contig.realize(contig, residuesInChain: residuesInChain, draw: draw)
        // The grammar accepts `R` and `D` suffixes; this port does not. Every token
        // it builds is `TokenType.PROTEIN`, so an RNA or DNA segment would be
        // silently designed as a peptide — a wrong answer rather than a refusal.
        // Parsing the suffix and then discarding it is what makes that possible, so
        // the refusal belongs here, where the protein assumption is actually made.
        for segment in segments {
            guard let type = segment.chainType, type != .polypeptide else { continue }
            throw TokenLayoutError.unsupportedChainType(type)
        }
        return TokenLayout(segments, pathLengthCap: pathLengthCap)
    }

    // MARK: - connectivity

    /// `[n][n]` shortest bond-graph hop count, clamped to the cutoff.
    ///
    /// Two tokens of different chains are **not connected**, and a disconnected pair
    /// is 0 — the same value as the self-diagonal, because the feature's bin 0 means
    /// "self, disconnected, or cross-group" alike.
    public func tokenBondPathLength() -> [[Int]] {
        let n = count
        var out = [[Int]](repeating: [Int](repeating: 0, count: n), count: n)
        for i in 0 ..< n {
            for j in 0 ..< n where tokens[i].chainIndex == tokens[j].chainIndex {
                out[i][j] = min(abs(i - j), pathLengthCap)
            }
        }
        return out
    }

    /// `[n][n]` backbone linkage direction: `+1` at `[downstream, upstream]`, `-1` at
    /// the transpose, `0` elsewhere. Signed, so it carries chain direction.
    public func canonicalPolymerOrdering() -> [[Int]] {
        let n = count
        var out = [[Int]](repeating: [Int](repeating: 0, count: n), count: n)
        for i in 1 ..< max(n, 1) where tokens[i].chainIndex == tokens[i - 1].chainIndex {
            out[i][i - 1] = 1
            out[i - 1][i] = -1
        }
        return out
    }

    /// Where one chain ends and the next begins, as token indices.
    public func chainBreaks() -> [Int] {
        guard count > 1 else { return [] }
        return (1 ..< count).filter { tokens[$0].chainIndex != tokens[$0 - 1].chainIndex }
    }

    /// Token ranges, one per output chain, in contig order.
    ///
    /// Written as "cut at each break" rather than as a scan with a trailing guard.
    /// The scan form needs `1 ... max(count, 1)` to cover the final range, and that
    /// upper bound is one past the end for an empty layout — a guard placed after
    /// the loop never runs.
    public func chainRanges() -> [Range<Int>] {
        guard count > 0 else { return [] }
        let cuts = [0] + chainBreaks() + [count]
        return zip(cuts, cuts.dropFirst()).map { $0 ..< $1 }
    }

    // MARK: - grouping ids

    /// One id per token, constant within a chain and distinct between chains.
    ///
    /// The three features that read these — index group, molecule id, molecule
    /// entity — are all **sameness** comparisons, `a[i] == a[j]`. Only the partition
    /// is observable, so the labels are a dense 0-based numbering rather than a
    /// reconstruction of upstream's internal ids. What must not happen is the
    /// de-novo mistake of leaving them at -1, which makes every token a molecule of
    /// its own.
    public func chainGroupIDs() -> [Int] { tokens.map(\.chainIndex) }

    /// `true` for each token that belongs to a designed segment.
    public func designedMask() -> [Bool] { tokens.map(\.isDesigned) }
}

public enum TokenLayoutError: Error, Equatable, CustomStringConvertible {
    case unsupportedChainType(Contig.SegmentChainType)

    public var description: String {
        switch self {
        case .unsupportedChainType(let t):
            let name = t == .rna ? "RNA" : "DNA"
            return """
                Contig asks for a \(name) segment (suffix '\(t.rawValue)'). This port \
                builds protein tokens only — the full 147-class vocabulary is not \
                ported yet, so a nucleic-acid segment would be designed as a peptide.
                """
        }
    }
}

/// The `C_CRDGRP` contract: which atom37 slots carry coordinate conditioning, and
/// which of them share a rigid group.
public enum CoordGroups {
    /// No coordinate conditioning of any kind.
    public static let unknown = -1
    /// Standard absolute-coordinate conditioning.
    public static let knownAbsolute = 0

    /// Relabel conditioned groups to `0 ..< k`, leaving `unknown` alone.
    ///
    /// Upstream's `CoordGroupSeqFeat` does **not** pass the raw ids to the network.
    /// It relabels them with a fresh random permutation of `0 ..< k` on every forward
    /// pass, so that nothing can be learned from a particular id and only which slots
    /// share a group remains observable. Feeding the raw ids instead presents a value
    /// the model never saw in training — for a one-group binder, `1` where every
    /// training example had `0`.
    ///
    /// Any permutation is as good as any other, so this is the identity one: ascending
    /// id order. For the single-group case it reproduces upstream exactly, because
    /// `randperm(1)` can only be `[0]`.
    public static func densified(_ groups: [Int]) -> [Int] {
        let distinct = Set(groups.filter { $0 != unknown }).sorted()
        var rank: [Int: Int] = [:]
        for (i, g) in distinct.enumerated() { rank[g] = i }
        return groups.map { $0 == unknown ? unknown : rank[$0]! }
    }
}
