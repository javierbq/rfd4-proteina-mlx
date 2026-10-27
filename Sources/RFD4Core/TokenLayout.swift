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
    public init(_ segments: [Contig.RealizedSegment], pathLengthCap: Int = 65) {
        var tokens: [Token] = []
        for (segmentIndex, segment) in segments.enumerated() {
            for offset in 0 ..< segment.length {
                tokens.append(Token(
                    chainIndex: segment.chainIndex,
                    chainID: segment.chainID,
                    segmentIndex: segmentIndex,
                    inputResidue: segment.preserved.map { $0.start + offset }))
            }
        }
        self.init(tokens: tokens, pathLengthCap: pathLengthCap)
    }

    /// Build from a contig string, drawing each expandable segment's length.
    public static func make(
        contig: String, draw: (Int, Int) -> Int, pathLengthCap: Int = 65
    ) throws -> TokenLayout {
        TokenLayout(try Contig.realize(contig, draw: draw), pathLengthCap: pathLengthCap)
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
        (1 ..< max(count, 1)).filter { tokens[$0].chainIndex != tokens[$0 - 1].chainIndex }
    }

    /// Token ranges, one per output chain, in contig order.
    public func chainRanges() -> [Range<Int>] {
        var out: [Range<Int>] = []
        var start = 0
        for i in 1 ... max(count, 1) {
            if i == count || tokens[i].chainIndex != tokens[start].chainIndex {
                out.append(start ..< i)
                start = i
            }
        }
        return count == 0 ? [] : out
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
