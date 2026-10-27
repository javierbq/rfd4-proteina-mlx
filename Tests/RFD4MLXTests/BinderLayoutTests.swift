import Foundation
import MLX
import RFD4Core
import Testing
@testable import RFD4MLX

/// The binder design path, against upstream's own shipped PD-L1 recipe.
///
/// `"50-120, /0, A17-131"` run through `ConditionSpecDataset` and the design
/// validation pipeline gives a 225-token batch: 110 designed residues, a chain
/// break, then 115 residues of the target. The fixture is that batch.
///
/// This is the test the de-novo suite could not be: every conditioning feature is
/// identically zero without a target, so the unconditioned fixtures pass whether the
/// conditioning is implemented or not, and the connectivity of a single chain hides
/// every mistake that only a chain *break* can expose.
@Suite("binder layout")
struct BinderLayoutTests {

    struct Fixture: Decodable, Sendable {
        struct Entry: Decodable, Sendable {
            let shape: [Int]
            let b64: String?
            let dtype: String
            let corner: Int?
        }
        struct Creator: Decodable, Sendable {
            let max: Double
            let dead: Bool
            /// Captured for the sequence track only; see `dump_binder_batch.py`.
            let output: Entry?
            let offset: Int?
        }
        let source: String
        let contig: String
        let seed: Int
        let nTokens: Int
        let seqProbKeys: [String]
        let fields: [String: Entry]
        let factories: [String: Entry]
        let creators: [String: Creator]
        enum CodingKeys: String, CodingKey {
            case source, contig, seed, fields, factories, creators
            case nTokens = "n_tokens"
            case seqProbKeys = "seq_prob_keys"
        }
    }

    nonisolated(unsafe) static let fixture: Fixture? = {
        guard let url = Bundle.module.url(forResource: "binder", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Fixture.self, from: data)
    }()

    static func array(_ e: Fixture.Entry) -> MLXArray? {
        guard let b64 = e.b64, let d = Data(base64Encoded: b64) else { return nil }
        return MLXArray(d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, e.shape)
    }

    static func field(_ k: String) throws -> MLXArray {
        let fx = try #require(fixture, "binder.json missing")
        let e = try #require(fx.fields[k], "no field \(k)")
        return try #require(array(e), "field \(k) was summarised, not stored")
    }

    /// The realized layout: 110 designed residues, then the target's 115.
    static func layout() -> TokenLayout {
        // The contig's expandable segment is 50-120; upstream's draw with seed 0 gave
        // 110, so the fixture is that problem. Pinning the draw here is what makes
        // the comparison a comparison.
        // The target chain is residues 17-131 with nothing unmodelled, which is why
        // this fixture's segment is its full arithmetic span; the residue source is
        // supplied rather than assumed because a chain with a gap would not be.
        try! TokenLayout.make(
            contig: "50-120, /0, A17-131",
            residuesInChain: { $0 == "A" ? Array(17 ... 131) : [] },
            draw: { _, _ in 110 })
    }

    // MARK: - contig

    @Test("the shipped binder contig parses into its three tokens")
    func parseBinderContig() throws {
        let tokens = try Contig.parse("50-120, /0, A17-131")
        #expect(tokens == [
            .expandable(Contig.ExpandableSegment(name: nil, min: 50, max: 120)),
            .chainBreak,
            .preserved(Contig.PreservedSegment(chainID: "A", start: 17, end: 131)),
        ])
    }

    @Test("token shapes are told apart the way upstream tells them apart")
    func tokenShapes() throws {
        // A leading letter makes it a preserved range; a leading digit makes the same
        // text a length range. `A1-10` and `1-10` differ only in that first character.
        #expect(try Contig.parseToken("A1-10")
                == .preserved(Contig.PreservedSegment(chainID: "A", start: 1, end: 10)))
        #expect(try Contig.parseToken("1-10")
                == .expandable(Contig.ExpandableSegment(name: nil, min: 1, max: 10)))
        // A colon makes it NAMED, and the name is whatever precedes the colon — which
        // may itself start with a letter, so the leading-character test must not run
        // first or `X1:5-8` reads as preserved chain "X" residue 1.
        #expect(try Contig.parseToken("X1:5-8")
                == .expandable(Contig.ExpandableSegment(name: "X1", min: 5, max: 8)))
        // A trailing P/R/D is a chain type, not part of the range.
        #expect(try Contig.parseToken("X1:5R")
                == .expandable(Contig.ExpandableSegment(name: "X1", min: 5, max: 5,
                                                        chainType: .rna)))
        #expect(try Contig.parseToken("/0") == .chainBreak)
        // A multi-character chain id takes the whole leading run of letters.
        #expect(try Contig.parseToken("AB12-20")
                == .preserved(Contig.PreservedSegment(chainID: "AB", start: 12, end: 20)))
    }

    @Test("chain ids are assigned run-first, not left to right")
    func chainRunFirst() throws {
        // Upstream's worked example. `E1` precedes `B1-10` in the string but shares a
        // run with it, so the run's id is only knowable once the whole run is read; a
        // single left-to-right walk gives `E1` a fresh chain instead of B.
        let runs = try Contig.chainRuns("A1-10, /0, E1:1-20, B1-10")
        #expect(runs.count == 2)
        #expect(runs[0].chainID == "A")
        #expect(runs[1].chainID == "B")
    }

    @Test("a wholly expandable run takes a fresh id, never a claimed one")
    func syntheticChainID() throws {
        let runs = try Contig.chainRuns("A1-5, /0, X1:5-8, X2:10")
        #expect(runs.count == 2)
        #expect(runs[1].chainID != "A")
        // Both segments of that run land on the one new chain.
        #expect(runs[1].tokens.count == 2)
    }

    /// Three details measured against upstream rather than inferred from its prose.
    @Test("the sentinel, whitespace and chain-type suffixes follow upstream")
    func upstreamDetails() throws {
        // `atomworks.constants.MASKED` is the literal "<M>". Checking for the word
        // "MASKED" gets it exactly backwards — it admits the sentinel upstream
        // rejects and rejects a name upstream allows.
        #expect(throws: ContigError.self) { try Contig.parseToken("<M>:5-8") }
        #expect(try Contig.parseToken("MASKED:5-8")
                == .expandable(Contig.ExpandableSegment(name: "MASKED", min: 5, max: 8)))

        // Python's str.strip() takes newlines; Swift's .whitespaces does not, and a
        // contig written across lines in a YAML block carries them.
        #expect(try Contig.parseToken("\n  A1-10 \t\r\n")
                == .preserved(Contig.PreservedSegment(chainID: "A", start: 1, end: 10)))
        let multiline = try Contig.parse("50-120,\n  /0,\n  A17-131")
        #expect(multiline.count == 3)

        // The grammar accepts R and D; this port builds protein tokens only, so it
        // must refuse rather than design a peptide and call it RNA.
        let source: (String) -> [Int] = { $0 == "A" ? Array(1 ... 200) : [] }
        #expect(throws: TokenLayoutError.self) {
            try TokenLayout.make(contig: "A1-10, /0, X1:5R",
                                 residuesInChain: source, draw: { lo, _ in lo })
        }
        #expect(throws: TokenLayoutError.self) {
            try TokenLayout.make(contig: "10D", residuesInChain: source, draw: { lo, _ in lo })
        }
        // An explicit P suffix is the thing it already does, so it passes.
        #expect(throws: Never.self) {
            try TokenLayout.make(contig: "10P", residuesInChain: source, draw: { lo, _ in lo })
        }
    }

    @Test("malformed contigs are rejected rather than half-read")
    func contigErrors() throws {
        #expect(throws: ContigError.self) { try Contig.parse("A10-5") }       // reversed
        #expect(throws: ContigError.self) { try Contig.parse("A1-10,,B1-5") } // stray comma
        #expect(throws: ContigError.self) { try Contig.chainRuns("A1-10, X1:5, X1:6") }
        // Two preserved chains in one run with no `/0` is ambiguous, not a merge.
        #expect(throws: ContigError.self) { try Contig.chainRuns("A1-10, B1-10") }
        // The same input residue may not be claimed twice.
        #expect(throws: ContigError.self) { try Contig.chainRuns("A1-10, /0, A5-15") }
    }

    // MARK: - layout against upstream's batch

    @Test("the fixture is the binder problem it claims to be")
    func fixtureShape() throws {
        let fx = try #require(Self.fixture, "binder.json missing")
        #expect(fx.nTokens == 225)
        #expect(fx.contig == "50-120, /0, A17-131")
        #expect(fx.seqProbKeys == DesignProblem.sequenceProbabilityChannels,
                "sequence-probability channel order differs from upstream's")
        let live = fx.creators.values.filter { !$0.dead }.count
        #expect(live == 16, "expected 16 live creators for a free-pose binder, got \(live)")
        // The absolute-coordinate feature is inert here and the internal one is not:
        // this recipe gives the target's shape and leaves its pose free.
        #expect(fx.creators["init_repr_factory/ConditionalAtom37NanometersCoorsSeqFeat"]?
            .dead == true)
        #expect(fx.creators[
            "init_repr_factory/ConditionalAtom37NanometersInternalCoorsSeqFeat"]?
            .dead == false)
    }

    @Test("the layout reproduces upstream's connectivity exactly")
    func connectivity() throws {
        let layout = Self.layout()
        #expect(layout.count == 225)
        #expect(layout.chainBreaks() == [110], "the chain break is not where upstream put it")
        #expect(layout.chainRanges() == [0 ..< 110, 110 ..< 225])

        let n = layout.count
        func mine(_ rows: [[Int]]) -> MLXArray {
            MLXArray(rows.flatMap { $0.map(Float.init) }, [1, n, n])
        }
        for (name, got, want) in [
            ("token_bond_path_length", mine(layout.tokenBondPathLength()),
             try Self.field("token_bond_path_length")),
            ("canonical_polymer_ordering", mine(layout.canonicalPolymerOrdering()),
             try Self.field("canonical_polymer_ordering")),
        ] {
            #expect(got.shape == want.shape, "\(name): \(got.shape) != \(want.shape)")
            guard got.shape == want.shape else { continue }
            let diff = MLX.abs(got - want).max().item(Float.self)
            #expect(diff == 0, "\(name): differs by \(diff)")
            if diff == 0 { print("ok   layout \(name): exact") }
        }
    }

    @Test("the grouping ids partition the tokens the way upstream's do")
    func grouping() throws {
        let layout = Self.layout()
        let mine = layout.chainGroupIDs()
        // The three features that read these compare a[i] == a[j], so only the
        // partition is observable — which is as well, since upstream's own chain_iid
        // labels are not stable between runs of the same spec. Comparing labels would
        // be comparing something the model cannot see.
        for key in ["molecule_iid", "molecule_entity_id", "condition_index-group_1_token"] {
            let theirs = try Self.field(key)[0].asArray(Float.self)
            #expect(theirs.count == mine.count)
            var agree = true
            for i in 0 ..< mine.count {
                for j in 0 ..< mine.count
                where (mine[i] == mine[j]) != (theirs[i] == theirs[j]) {
                    agree = false
                }
                if !agree { break }
            }
            #expect(agree, "\(key) partitions the tokens differently")
        }
    }

    @Test("both chains are generated — the target is not frozen")
    func emitMasks() throws {
        // `bb_ca` emits where padding_mask & ~is_in_context_conditioning, and upstream's
        // binder batch has that second array all false. The target is held in place by
        // its conditioning, not by a mask. A port that froze the target would be
        // solving a different problem from the one the weights were trained for.
        let context = try Self.field("is_in_context_conditioning")
        #expect(MLX.all(context .== 0).item(Bool.self),
                "upstream freezes some tokens; the port's emit masks must follow")
        let padding = try Self.field("padding_mask")
        #expect(MLX.all(padding .== 1).item(Bool.self))
    }
}
