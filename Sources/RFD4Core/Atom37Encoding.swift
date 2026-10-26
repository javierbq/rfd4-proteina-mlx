import Foundation

/// A structure in RFD4's atom37 encoding: one row per token, 37 atom slots per row.
///
/// Counterpart of what `atomworks.ml.transforms.encoding.atom_array_to_encoding`
/// returns — the `xyz` / `mask` / `seq` triple the model consumes.
public struct Atom37Encoding: Sendable {
    /// `[nTokens][37]` coordinates. Unoccupied slots are `.zero` and masked off.
    public var xyz: [[Vec3]]
    /// `[nTokens][37]` occupancy — true where an atom was actually resolved.
    public var mask: [[Bool]]
    /// `[nTokens]` residue class indices into the vocabulary.
    public var seq: [Int]

    public var tokenCount: Int { seq.count }
}

public enum Atom37EncodingError: Error, CustomStringConvertible {
    case unknownResidue(String)
    case unknownAtom(residue: String, atom: String)
    case sequenceOutOfRange(Int)

    public var description: String {
        switch self {
        case .unknownResidue(let r):
            "residue \(r) is not in the vocabulary"
        case .unknownAtom(let r, let a):
            "atom \(a) has no slot in residue \(r)"
        case .sequenceOutOfRange(let i):
            "sequence index \(i) is outside the vocabulary"
        }
    }
}

extension AtomArray {
    /// Encode to atom37.
    ///
    /// Tokens are taken from `tokIdx`. An atom whose residue/atom pair has no slot
    /// is an error rather than a silent drop: upstream tolerates -1 slots further up
    /// the pipeline, but at this boundary a missing slot means the caller built an
    /// `AtomArray` the vocabulary cannot represent, and dropping it would hand the
    /// model a structure that is quietly not the one supplied.
    public func atom37Encoding(
        _ vocabulary: Vocabulary = .unifiedAtom37
    ) throws -> Atom37Encoding {
        let starts = tokenStarts()
        let slots = atom37Slots(vocabulary)
        let bounds = zip(starts, starts.dropFirst() + [count])

        var xyz = [[Vec3]](), mask = [[Bool]](), seq = [Int]()
        xyz.reserveCapacity(starts.count)
        mask.reserveCapacity(starts.count)
        seq.reserveCapacity(starts.count)

        for (start, end) in bounds {
            let residue = resName[start]
            guard let classIndex = vocabulary.index(of: residue) else {
                throw Atom37EncodingError.unknownResidue(residue)
            }
            var rowXYZ = [Vec3](repeating: .zero, count: Vocabulary.slotCount)
            var rowMask = [Bool](repeating: false, count: Vocabulary.slotCount)
            for i in start..<end {
                let slot = slots[i]
                guard slot >= 0 else {
                    throw Atom37EncodingError.unknownAtom(residue: resName[i], atom: atomName[i])
                }
                rowXYZ[slot] = coord[i]
                rowMask[slot] = true
            }
            xyz.append(rowXYZ)
            mask.append(rowMask)
            seq.append(classIndex)
        }
        return Atom37Encoding(xyz: xyz, mask: mask, seq: seq)
    }

    /// Decode from atom37 — the inverse of `atom37Encoding(_:)`.
    ///
    /// Only masked-in slots become atoms, so the result carries exactly the atoms
    /// that were resolved. Atom and residue names come from the vocabulary layout,
    /// which is what makes the round trip name-stable.
    public static func from(
        _ encoding: Atom37Encoding,
        vocabulary: Vocabulary = .unifiedAtom37
    ) throws -> AtomArray {
        var resNames = [String](), atomNames = [String](), coords = [Vec3]()
        var tokIdx = [Int]()

        for (token, classIndex) in encoding.seq.enumerated() {
            guard vocabulary.classes.indices.contains(classIndex) else {
                throw Atom37EncodingError.sequenceOutOfRange(classIndex)
            }
            let residue = vocabulary.classes[classIndex]
            for slot in 0..<Vocabulary.slotCount where encoding.mask[token][slot] {
                let atom = residue.atoms[slot].trimmingCharacters(in: .whitespaces)
                guard !atom.isEmpty else {
                    throw Atom37EncodingError.unknownAtom(residue: residue.name, atom: "slot \(slot)")
                }
                resNames.append(residue.name)
                atomNames.append(atom)
                coords.append(encoding.xyz[token][slot])
                tokIdx.append(token)
            }
        }

        var out = AtomArray(resName: resNames, atomName: atomNames, coord: coords)
        out.tokIdx = tokIdx
        return out
    }
}
