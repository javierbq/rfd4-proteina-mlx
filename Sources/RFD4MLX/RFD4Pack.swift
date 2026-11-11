// RFD4Pack.swift — the installable, versioned weight bundle.
//
// `Weights.load` takes a path to one safetensors file. RayMol hands a model a
// DIRECTORY, because its weight cache extracts a downloaded zip to
// `<root>/<id>/<version>/` and passes that path along. So shipping inside RayMol
// needs a directory-shaped entry point no matter what precision the weights are
// in, and that entry point is the natural place to put the things a bare
// safetensors file cannot carry: what checkpoint it came from, what precision it
// is in, and whether it still matches the vocabulary this build was compiled with.
//
// The layout is flat by requirement, not by taste. RayMol's `WeightCache._extract`
// compares the zip's member set against the bundle's declared `members` and
// rejects any member that is not a plain file name, so a pack can never grow a
// subdirectory.
import CryptoKit
import Foundation
import MLX
import RFD4Core

public enum RFD4PackError: Error, CustomStringConvertible, LocalizedError {
    case missingManifest(URL)
    case missingFile(String)
    case checksumMismatch(file: String, expected: String, got: String)
    case unsupportedFormat(String)
    case unknownPrecision(String)
    case vocabularyMismatch(packName: String, packSnapshot: String,
                            buildName: String, buildSnapshot: String)
    case unsupportedQuantization(String)

    public var description: String {
        switch self {
        case .missingManifest(let u):
            "no manifest.json in \(u.path)"
        case .missingFile(let f):
            "pack file missing: \(f)"
        case .checksumMismatch(let f, let e, let g):
            "sha256 mismatch for \(f): expected \(e), got \(g)"
        case .unsupportedFormat(let v):
            "unsupported pack format version: \(v) (this build reads \(RFD4Pack.supportedFormat))"
        case .unknownPrecision(let p):
            "pack declares precision '\(p)', which this build does not know"
        case .vocabularyMismatch(let pn, let ps, let bn, let bs):
            """
            pack was exported against vocabulary '\(pn)' (CCD \(ps)) but this build \
            compiled in '\(bn)' (CCD \(bs)). The sequence head's output classes are \
            positional, so a mismatch renames residues silently rather than failing.
            """
        case .unsupportedQuantization(let m):
            "pack is quantized (\(m)); this build has only a dense matmul path"
        }
    }

    /// A host that surfaces `localizedDescription` must get the sha256 that
    /// mismatched or the vocabulary that moved, not a generic failure sentence.
    public var errorDescription: String? { description }
}

/// A validated RFD4 weight pack: manifest, sha256-checked weights, loaded into MLX.
public struct RFD4Pack {

    /// Where the weights came from, so a pack can never be of unknown origin.
    ///
    /// RFD3's pack format learned this the hard way: its checkpoints carry both a
    /// raw and an EMA copy of every tensor, inference runs the EMA one, and a pack
    /// built from the wrong branch is a different model that looks identical. RFD4's
    /// checkpoint has no such twin, but the general lesson holds — the cost of
    /// recording provenance is one JSON object, and the cost of not recording it is
    /// a question nobody can answer later.
    public struct Provenance: Codable, Sendable {
        public let sourceCheckpoint: String
        public let sourceCheckpointSHA256: String
        public let exportedAt: String
        public let exportedBy: String

        public init(sourceCheckpoint: String, sourceCheckpointSHA256: String,
                    exportedAt: String, exportedBy: String) {
            self.sourceCheckpoint = sourceCheckpoint
            self.sourceCheckpointSHA256 = sourceCheckpointSHA256
            self.exportedAt = exportedAt
            self.exportedBy = exportedBy
        }
    }

    /// The vocabulary the weights were exported against.
    ///
    /// `vocabulary.json` is compiled into RFD4Core as a resource, but it is really
    /// part of the weights: the sequence head emits one logit per residue class, and
    /// the classes are identified by POSITION, not by name. Re-generating the
    /// vocabulary from a newer CCD snapshot can reorder or insert a class, after
    /// which every residue downstream of the change is named wrong — and nothing
    /// about that failure looks like a failure. It produces a plausible structure
    /// with the wrong chemistry.
    ///
    /// So the pack records what it was built against and the loader refuses a
    /// mismatch. This costs three strings and removes an entire class of silent
    /// corruption.
    public struct VocabularyPin: Codable, Sendable {
        public let name: String
        public let ccdSnapshot: String
        public let classCount: Int

        public init(name: String, ccdSnapshot: String, classCount: Int) {
            self.name = name
            self.ccdSnapshot = ccdSnapshot
            self.classCount = classCount
        }
    }

    /// Reserved. Absent (nil) means dense weights, which is the only thing this
    /// build can execute.
    ///
    /// It is declared now, unused, so that adding an int8 pack later is a `format`
    /// bump against a schema that already has somewhere to put the group size —
    /// rather than a new schema, which would strand every pack already installed.
    /// `AffineLinear` in boltz-mlx is the shape this would take: `bits`,
    /// `groupSize`, `mode`, plus per-tensor logical/physical widths.
    public struct Quantization: Codable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public let mode: String

        public init(bits: Int, groupSize: Int, mode: String) {
            self.bits = bits
            self.groupSize = groupSize
            self.mode = mode
        }
    }

    public struct Manifest: Codable, Sendable {
        public let name: String
        public let version: String
        /// Pack format version. Probed before the full decode so an old pack fails
        /// with "unsupported format" rather than a confusing missing-key error.
        public let format: String
        /// "fp32" | "fp16" | "bf16" — the dtype the weights are stored in.
        public let precision: String
        public let weights: Provenance
        public let vocabulary: VocabularyPin
        /// filename -> sha256 hex, for every file in the pack except the manifest.
        public let files: [String: String]
        public let quantization: Quantization?

        public init(name: String, version: String, format: String, precision: String,
                    weights: Provenance, vocabulary: VocabularyPin,
                    files: [String: String], quantization: Quantization? = nil) {
            self.name = name
            self.version = version
            self.format = format
            self.precision = precision
            self.weights = weights
            self.vocabulary = vocabulary
            self.files = files
            self.quantization = quantization
        }
    }

    /// Just enough to reject an unreadable format with a clear message.
    private struct FormatProbe: Codable { let format: String }

    public static let supportedFormat = "1"
    public static let weightsFilename = "rfd4_core.safetensors"
    public static let manifestFilename = "manifest.json"

    public let directory: URL
    public let manifest: Manifest
    public let weights: Weights

    /// The precision the pack is stored in.
    public var precision: Weights.Precision {
        Weights.Precision(manifestName: manifest.precision) ?? .float32
    }

    /// - Parameter verifyChecksums: hashing the weights costs about a second per
    ///   gigabyte and happens once per process. Leave it on. A truncated download
    ///   that gets past RayMol's zip-level digest surfaces here as a named file with
    ///   a named mismatch; without it, it surfaces as a missing-key crash deep in a
    ///   layer constructor, which is a far worse thing to debug from a user report.
    public init(directory: URL, verifyChecksums: Bool = true) throws {
        self.directory = directory

        let manifestURL = directory.appendingPathComponent(Self.manifestFilename)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw RFD4PackError.missingManifest(directory)
        }
        let data = try Data(contentsOf: manifestURL)

        // Format first, whole manifest second. A future format that adds a required
        // key would otherwise be reported as that key being absent, which points the
        // reader at the wrong problem.
        let probe = try JSONDecoder().decode(FormatProbe.self, from: data)
        guard probe.format == Self.supportedFormat else {
            throw RFD4PackError.unsupportedFormat(probe.format)
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        self.manifest = manifest

        if let q = manifest.quantization {
            throw RFD4PackError.unsupportedQuantization(
                "\(q.bits)-bit \(q.mode), group \(q.groupSize)")
        }
        guard let precision = Weights.Precision(manifestName: manifest.precision) else {
            throw RFD4PackError.unknownPrecision(manifest.precision)
        }

        for (name, expected) in manifest.files {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw RFD4PackError.missingFile(name)
            }
            guard verifyChecksums else { continue }
            let got = try Self.sha256Hex(of: url)
            guard got == expected else {
                throw RFD4PackError.checksumMismatch(file: name, expected: expected, got: got)
            }
        }

        let vocabulary = Vocabulary.unifiedAtom37
        let pin = manifest.vocabulary
        guard pin.name == vocabulary.name,
              pin.ccdSnapshot == vocabulary.ccdSnapshot,
              pin.classCount == vocabulary.tokenCount
        else {
            throw RFD4PackError.vocabularyMismatch(
                packName: pin.name, packSnapshot: pin.ccdSnapshot,
                buildName: vocabulary.name, buildSnapshot: vocabulary.ccdSnapshot)
        }

        // The pack's declared precision wins. `RFD4_DTYPE` means "narrow this fp32
        // checkpoint on the way in", and applying it to a pack already stored in
        // half would be a no-op at best; letting it widen one is impossible, since
        // the bits are gone. `Weights.load` with the pack's own precision is
        // therefore the identity for a half pack and the documented cast for fp32.
        self.weights = try Weights.load(
            directory.appendingPathComponent(Self.weightsFilename), precision: precision)
    }

    /// Streamed sha256, so a gigabyte of weights is never fully resident just to
    /// hash it.
    public static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while case let chunk = try handle.read(upToCount: 8 << 20) ?? Data(), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

extension Weights {
    /// Weights from either a pack directory or a bare `.safetensors` file.
    ///
    /// Hosts hand over a directory — RayMol's weight cache extracts a bundle to
    /// `<root>/<id>/<version>/` and passes that path along — while the CLI and the
    /// test suite are usually pointed straight at a checkpoint. Deciding by what is
    /// actually on disk means no caller has to know which it is holding.
    ///
    /// A directory that is not a pack fails AS A PACK, naming the missing manifest,
    /// rather than as an unreadable file. That matters: "no manifest.json in
    /// /path/to/dir" tells someone what to do next and "could not read weights"
    /// does not.
    public static func loadPathOrPack(_ path: String) throws -> Weights {
        let expanded = (path as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory)
        else { throw WeightsError.missing(expanded) }
        if isDirectory.boolValue {
            return try RFD4Pack(directory: URL(fileURLWithPath: expanded)).weights
        }
        return try Weights.load(URL(fileURLWithPath: expanded))
    }
}
