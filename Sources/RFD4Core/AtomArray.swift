import Foundation

public struct Vec3: Equatable, Sendable {
    public var x: Float, y: Float, z: Float
    public init(_ x: Float, _ y: Float, _ z: Float) { (self.x, self.y, self.z) = (x, y, z) }
    public static let zero = Vec3(0, 0, 0)
}

/// Chain classification, mirroring `atomworks.enums.ChainType` for the cases the
/// port needs. Carried only so that tests can assert slot assignment *ignores* it.
public enum ChainType: Int, Sendable {
    case nonPolymer = 0
    case polypeptideL = 1
    case polyribonucleotide = 2
    case polydeoxyribonucleotide = 3
}

/// A flat, struct-of-arrays stand-in for `biotite.structure.AtomArray`.
///
/// Only the annotations this port reads are modelled. Biotite stores annotations
/// in a dynamic dictionary; here they are stored fields, because every annotation
/// the featurizer consumes is known statically and a typo in an annotation name
/// should be a compile error rather than a silent zero-fill.
///
/// Optional annotations are `nil` when absent, which is how `build_atom37_slots`
/// distinguishes "no `atomize` annotation" from "`atomize` is all false".
public struct AtomArray: Sendable {
    public var resName: [String]
    public var atomName: [String]
    public var coord: [Vec3]

    public var element: [String]?
    public var tokIdx: [Int]?
    public var chainID: [String]?
    public var resID: [Int]?
    public var occupancy: [Float]?
    public var bFactor: [Float]?
    public var chainType: [ChainType]?

    /// Explicit atomization. Takes effect only when `tokenType` is absent.
    public var atomize: [Bool]?
    /// Per-atom token type; when present it wins over `atomize`
    /// (`utils/token_encoding.py:build_atom37_slots`).
    public var tokenType: [Int]?

    public init(resName: [String], atomName: [String], coord: [Vec3]) {
        precondition(resName.count == atomName.count && resName.count == coord.count,
                     "AtomArray columns must be the same length")
        self.resName = resName
        self.atomName = atomName
        self.coord = coord
    }

    /// An array of `count` atoms with empty names and zeroed coordinates, matching
    /// `AtomArray(length:)` in biotite.
    public init(length: Int) {
        self.init(resName: [String](repeating: "", count: length),
                  atomName: [String](repeating: "", count: length),
                  coord: [Vec3](repeating: .zero, count: length))
    }

    public var count: Int { resName.count }

    /// Valid atom indices. `AtomArray` is a struct of columns rather than a
    /// collection of atoms, so this is the way to iterate positions.
    public var indices: Range<Int> { 0 ..< count }

    /// Vocabulary slot per atom, `-1` where the residue/atom pair is unknown.
    ///
    /// Mirrors `build_atom37_slots`: lookup is residue-qualified and independent of
    /// chain classification, and explicitly atomized atoms are forced to slot 1
    /// regardless of what their residue layout says.
    public func atom37Slots(_ vocabulary: Vocabulary = .unifiedAtom37) -> [Int] {
        let atomized: [Bool]
        if let tokenType {
            atomized = tokenType.map { $0 == TokenType.atomized }
        } else if let atomize {
            atomized = atomize
        } else {
            atomized = [Bool](repeating: false, count: count)
        }

        return (0..<count).map { i in
            if atomized[i] { return 1 }
            return vocabulary.slot(residue: resName[i], atom: atomName[i]) ?? -1
        }
    }

    /// Start index of each token, derived from `tokIdx`. Falls back to treating the
    /// whole array as one token when the annotation is absent.
    public func tokenStarts() -> [Int] {
        guard let tokIdx, !tokIdx.isEmpty else { return count > 0 ? [0] : [] }
        var starts = [0]
        for i in 1..<tokIdx.count where tokIdx[i] != tokIdx[i - 1] {
            starts.append(i)
        }
        return starts
    }
}
