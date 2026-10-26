import Foundation

/// Token-type ids, as carried by the `token_type` annotation and feature.
///
/// Transcribed from `rfproteina/encoding.py:11` and `datapipes/schema/proteina.py:40`.
/// Note `mask` and `unknown` deliberately share id 4 upstream — the mask token is
/// typed as unknown, not as a type of its own.
public enum TokenType {
    public static let protein = 0
    public static let rna = 1
    public static let dna = 2
    public static let atomized = 3
    public static let unknown = 4
    public static let glycan = 5
    public static let ncaa = 6

    /// Tensor row count for direct id indexing: gaps leave unused rows, aliases share a row.
    public static let count = 7
}

/// One of the 147 residue classes, in checkpoint order.
public struct ResidueClass: Sendable {
    public let name: String
    public let group: String
    public let tokenTypeID: Int
    /// Exactly 37 entries; `""` marks an empty slot.
    public let atoms: [String]
}

/// RFD4's 37-slot residue vocabulary — the `UNIFIED_ATOM37_ENCODING` of
/// `rfproteina/encoding.py`.
///
/// **This file is part of the weights.** The upstream YAML carries the warning
/// verbatim: *"Checkpoints depend on the residue order and each residue's atom-slot
/// order. Reordering either changes what the learned weights mean."* The JSON
/// resource is a transcription of that YAML by `src/rfd4_mlx_export/gen_vocabulary.py`
/// and must never be hand-edited.
public struct Vocabulary: Sendable {
    public static let slotCount = 37

    public let name: String
    public let ccdSnapshot: String
    public let classes: [ResidueClass]

    private let indexByName: [String: Int]
    private let slotByResidueAtom: [String: Int]

    /// Residue classes the autoencoder represents with local latents —
    /// everything but the atomized and unknown placeholders
    /// (`utils/constants.py:7`).
    public static let nonLocalLatentTokens: Set<String> = ["<A>", "<G>"]
    public static let maskedToken = "<M>"
    public static let unknownAA = "UNK"

    /// The canonical twenty. `UNK` is excluded, matching atomworks' `STANDARD_AA`.
    public static let standardAA: Set<String> = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]

    // MARK: - Loading

    public static let unifiedAtom37: Vocabulary = {
        guard let url = Bundle.module.url(forResource: "vocabulary", withExtension: "json") else {
            fatalError("vocabulary.json missing from RFD4Core resources")
        }
        do {
            return try Vocabulary(contentsOf: url)
        } catch {
            fatalError("vocabulary.json failed to load: \(error)")
        }
    }()

    public init(contentsOf url: URL) throws {
        let payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: url))
        guard payload.slots == Self.slotCount else {
            throw VocabularyError.badSlotCount(payload.slots)
        }

        name = payload.name
        ccdSnapshot = payload.ccdSnapshot
        classes = payload.classes.map {
            ResidueClass(name: $0.name, group: $0.group,
                         tokenTypeID: $0.tokenTypeId, atoms: $0.atoms)
        }

        var byName = [String: Int](minimumCapacity: classes.count)
        for (i, c) in classes.enumerated() {
            guard c.atoms.count == Self.slotCount else {
                throw VocabularyError.badLayout(c.name, c.atoms.count)
            }
            let occupied = c.atoms.filter { !$0.isEmpty }
            guard Set(occupied).count == occupied.count else {
                throw VocabularyError.repeatedAtom(c.name)
            }
            guard byName.updateValue(i, forKey: c.name) == nil else {
                throw VocabularyError.duplicateResidue(c.name)
            }
        }
        indexByName = byName

        // Residue-qualified keys, not bare atom names: purines and pyrimidines
        // reuse N1/C2/N3 and place them in different slots
        // (`utils/token_encoding.py:_build_atom37_lookups`).
        var slots = [String: Int](minimumCapacity: 4096)
        for c in classes where !Self.nonLocalLatentTokens.contains(c.name) && c.name != Self.maskedToken {
            for (slot, raw) in c.atoms.enumerated() {
                let atom = raw.trimmingCharacters(in: .whitespaces)
                guard !atom.isEmpty else { continue }
                slots["\(c.name)|\(atom)"] = slot
                // Standard proteins share slots, including atoms observed in UNK residues.
                if Self.standardAA.contains(c.name) {
                    slots["\(Self.unknownAA)|\(atom)"] = slot
                }
            }
        }
        slotByResidueAtom = slots
    }

    // MARK: - Lookup

    public var tokenCount: Int { classes.count }
    public var tokens: [String] { classes.map(\.name) }

    public func index(of residue: String) -> Int? { indexByName[residue] }

    public func atoms(of residue: String) -> [String]? {
        indexByName[residue].map { classes[$0].atoms }
    }

    /// Vocabulary slot for one residue/atom pair, or `nil` when the pair is unknown.
    /// Upstream returns -1 here; `AtomArray` reintroduces that at the array boundary.
    public func slot(residue: String, atom: String) -> Int? {
        slotByResidueAtom["\(residue)|\(atom.trimmingCharacters(in: .whitespaces))"]
    }

    // MARK: - Derived masks

    /// Legal atom37 slots per residue class, `[nTokens][37]`
    /// (`utils/constants.py:_build_unified_atom37_mask`).
    public func residueAtomMask() -> [[Bool]] {
        classes.map { c in
            c.atoms.map { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        }
    }

    /// Residue classes each token type may decode to, `[7][nTokens]`
    /// (`partial_autoencoder/decoder.py:get_allowed_residue_type_mask`).
    ///
    /// Groups whose id is `TokenType.unknown` — that is both `mask` and `unknown` —
    /// are skipped in the first pass, then row 4 is set to the union of all others.
    public func allowedResidueTypeMask() -> [[Bool]] {
        var allowed = [[Bool]](repeating: [Bool](repeating: false, count: tokenCount),
                               count: TokenType.count)
        for (i, c) in classes.enumerated() where c.tokenTypeID != TokenType.unknown {
            allowed[c.tokenTypeID][i] = true
        }
        for i in 0..<tokenCount {
            allowed[TokenType.unknown][i] = allowed.indices.contains { row in
                row != TokenType.unknown && allowed[row][i]
            }
        }
        return allowed
    }

    // MARK: - Decoding support

    private struct Payload: Decodable {
        struct Entry: Decodable {
            let name: String
            let group: String
            let tokenTypeId: Int
            let atoms: [String]
        }
        let name: String
        let ccdSnapshot: String
        let slots: Int
        let classes: [Entry]
    }
}

public enum VocabularyError: Error, CustomStringConvertible {
    case badSlotCount(Int)
    case badLayout(String, Int)
    case repeatedAtom(String)
    case duplicateResidue(String)

    public var description: String {
        switch self {
        case .badSlotCount(let n): "vocabulary declares \(n) slots, expected 37"
        case .badLayout(let r, let n): "\(r) has \(n) slots, expected 37"
        case .repeatedAtom(let r): "\(r) repeats an atom name"
        case .duplicateResidue(let r): "\(r) appears in more than one group"
        }
    }
}
