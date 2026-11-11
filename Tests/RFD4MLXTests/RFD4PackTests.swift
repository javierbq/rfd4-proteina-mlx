import Foundation
import MLX
import RFD4Core
import Testing

@testable import RFD4MLX

/// The pack is the thing standing between a downloaded zip and the network, so
/// what is asserted here is mostly what it REFUSES. A pack that loads slightly
/// wrong weights is worse than one that fails: the former produces a plausible
/// structure with the wrong chemistry.
///
/// No checkpoint needed — a handful of synthetic tensors exercises every path.
@Suite("rfd4 pack")
struct RFD4PackTests {

    static func scratchDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rfd4pack-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A pack with two float tensors and one integer one, written the way the
    /// exporter writes it.
    @discardableResult
    static func writePack(
        at directory: URL,
        precision: Weights.Precision = .float16,
        format: String = RFD4Pack.supportedFormat,
        vocabulary: RFD4Pack.VocabularyPin? = nil,
        quantization: RFD4Pack.Quantization? = nil,
        corruptChecksum: Bool = false
    ) throws -> [String: MLXArray] {
        let arrays: [String: MLXArray] = [
            "nn.a.weight": MLXArray(converting: [1.5, -2.25, 0.125, 7.0], [2, 2]),
            "nn.b.bias": MLXArray(converting: [0.5, -0.5]),
            "nn.lookup": MLXArray([3, 1, 4, 1], [4]),
        ]
        let narrowed = Weights.narrowed(arrays, to: precision)
        MLX.eval(Array(narrowed.values))
        let weightsURL = directory.appendingPathComponent(RFD4Pack.weightsFilename)
        try MLX.save(arrays: narrowed, metadata: ["format": "pt"], url: weightsURL)

        let sha = corruptChecksum
            ? String(repeating: "0", count: 64)
            : try RFD4Pack.sha256Hex(of: weightsURL)
        let vocab = Vocabulary.unifiedAtom37
        let manifest = RFD4Pack.Manifest(
            name: "RFD4-Proteina", version: "test", format: format,
            precision: precision.manifestName,
            weights: RFD4Pack.Provenance(
                sourceCheckpoint: "model.safetensors",
                sourceCheckpointSHA256: String(repeating: "a", count: 64),
                exportedAt: "2026-01-01T00:00:00Z", exportedBy: "test"),
            vocabulary: vocabulary ?? RFD4Pack.VocabularyPin(
                name: vocab.name, ccdSnapshot: vocab.ccdSnapshot,
                classCount: vocab.tokenCount),
            files: [RFD4Pack.weightsFilename: sha],
            quantization: quantization)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest)
            .write(to: directory.appendingPathComponent(RFD4Pack.manifestFilename))
        return arrays
    }

    @Test("a pack round-trips its tensors, narrowing floats and leaving integers")
    func roundTrip() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let original = try Self.writePack(at: dir, precision: .float16)

        let pack = try RFD4Pack(directory: dir)
        #expect(pack.precision == .float16)
        #expect(pack.weights.arrays.count == original.count)

        // The float tensors are half; the integer one is untouched. Casting an
        // index array to half would silently round anything past 2048.
        #expect(pack.weights.arrays["nn.a.weight"]?.dtype == .float16)
        #expect(pack.weights.arrays["nn.b.bias"]?.dtype == .float16)
        #expect(pack.weights.arrays["nn.lookup"]?.dtype == original["nn.lookup"]?.dtype)

        // These values are all exactly representable in fp16, so the round trip is
        // exact and any difference is a real defect rather than rounding.
        let got = pack.weights.arrays["nn.a.weight"]!.asType(.float32)
        let want = original["nn.a.weight"]!
        #expect(MLX.abs(got - want).max().item(Float.self) == 0)
    }

    @Test("an fp32 pack stays fp32 regardless of RFD4_DTYPE")
    func fp32PackIgnoresEnvironment() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writePack(at: dir, precision: .float32)
        // `Weights.load`'s default argument reads the environment; the pack passes
        // its own declared precision instead, so what is on disk is what is loaded.
        let pack = try RFD4Pack(directory: dir)
        #expect(pack.precision == .float32)
        #expect(pack.weights.arrays["nn.a.weight"]?.dtype == .float32)
    }

    @Test("a corrupt weights file is refused by name")
    func checksumMismatchIsRefused() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writePack(at: dir, corruptChecksum: true)
        #expect(throws: RFD4PackError.self) { try RFD4Pack(directory: dir) }
        // ...and the error has to name the file, because a user reporting this is
        // reading `localizedDescription` off a dialog and nothing else.
        do {
            _ = try RFD4Pack(directory: dir)
        } catch let error as RFD4PackError {
            #expect(error.description.contains(RFD4Pack.weightsFilename))
            #expect(error.errorDescription == error.description)
        }
    }

    @Test("checksum verification can be skipped, and then a corrupt pack loads")
    func checksumsCanBeSkipped() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writePack(at: dir, corruptChecksum: true)
        // Asserted so the cost of `verifyChecksums: false` is written down: it is
        // the difference between a named mismatch and no check at all.
        let pack = try RFD4Pack(directory: dir, verifyChecksums: false)
        #expect(pack.weights.arrays.count == 3)
    }

    @Test("a future pack format is refused before anything else is decoded")
    func unsupportedFormatIsRefused() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writePack(at: dir, format: "99")
        do {
            _ = try RFD4Pack(directory: dir)
            Issue.record("a format-99 pack loaded")
        } catch let error as RFD4PackError {
            #expect(error.description.contains("99"))
            #expect(error.description.contains("unsupported pack format"))
        }
    }

    @Test("a pack built against another vocabulary is refused")
    func vocabularyMismatchIsRefused() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        // A newer CCD snapshot can reorder or insert a residue class. The sequence
        // head's outputs are positional, so the result is correctly-shaped,
        // confidently-wrong chemistry — which is why this is an error and not a
        // warning.
        try Self.writePack(at: dir, vocabulary: RFD4Pack.VocabularyPin(
            name: Vocabulary.unifiedAtom37.name,
            ccdSnapshot: "2027-06-01",
            classCount: Vocabulary.unifiedAtom37.tokenCount))
        do {
            _ = try RFD4Pack(directory: dir)
            Issue.record("a pack with a different CCD snapshot loaded")
        } catch let error as RFD4PackError {
            #expect(error.description.contains("2027-06-01"))
            #expect(error.description.contains(Vocabulary.unifiedAtom37.ccdSnapshot))
        }
    }

    @Test("a class-count change is refused even at the same snapshot date")
    func vocabularyClassCountIsChecked() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vocab = Vocabulary.unifiedAtom37
        try Self.writePack(at: dir, vocabulary: RFD4Pack.VocabularyPin(
            name: vocab.name, ccdSnapshot: vocab.ccdSnapshot,
            classCount: vocab.tokenCount + 1))
        #expect(throws: RFD4PackError.self) { try RFD4Pack(directory: dir) }
    }

    @Test("a quantized pack is refused while there is no quantized matmul path")
    func quantizedPackIsRefused() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        // The field exists so int8 is a `format` bump rather than a new schema. Until
        // the matmul path exists, declaring it must fail loudly — silently running
        // dense matmuls over packed uint32 would produce noise, not an error.
        try Self.writePack(at: dir, quantization: RFD4Pack.Quantization(
            bits: 8, groupSize: 64, mode: "affine"))
        do {
            _ = try RFD4Pack(directory: dir)
            Issue.record("a quantized pack loaded")
        } catch let error as RFD4PackError {
            #expect(error.description.contains("8-bit affine"))
        }
    }

    @Test("a missing manifest names the directory it looked in")
    func missingManifestIsRefused() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try RFD4Pack(directory: dir)
            Issue.record("an empty directory loaded as a pack")
        } catch let error as RFD4PackError {
            #expect(error.description.contains(dir.path))
        }
    }

    @Test("a manifest naming an absent file names that file")
    func missingWeightsFileIsRefused() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writePack(at: dir)
        try FileManager.default.removeItem(
            at: dir.appendingPathComponent(RFD4Pack.weightsFilename))
        do {
            _ = try RFD4Pack(directory: dir)
            Issue.record("a pack with no weights file loaded")
        } catch let error as RFD4PackError {
            #expect(error.description.contains(RFD4Pack.weightsFilename))
        }
    }

    @Test("a path resolves as a pack or a bare checkpoint by what is on disk")
    func pathOrPackDispatch() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writePack(at: dir, precision: .float16)

        // A directory goes through the pack, so its declared precision applies.
        let fromPack = try Weights.loadPathOrPack(dir.path)
        #expect(fromPack.arrays["nn.a.weight"]?.dtype == .float16)

        // The same bytes named directly are a bare checkpoint: already fp16 on
        // disk, and `narrowed` leaves non-fp32 tensors alone whatever is asked.
        let fromFile = try Weights.loadPathOrPack(
            dir.appendingPathComponent(RFD4Pack.weightsFilename).path)
        #expect(fromFile.arrays.count == fromPack.arrays.count)

        // A directory that is not a pack fails AS A PACK. "no manifest.json in
        // <dir>" says what to do next; "could not read weights" does not.
        let empty = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: empty) }
        do {
            _ = try Weights.loadPathOrPack(empty.path)
            Issue.record("an empty directory loaded")
        } catch let error as RFD4PackError {
            #expect(error.description.contains("manifest.json"))
        }

        #expect(throws: (any Error).self) {
            try Weights.loadPathOrPack(dir.appendingPathComponent("nope").path)
        }
    }

    @Test("both manifest spellings of a precision parse, and round-trip")
    func precisionNames() {
        #expect(Weights.Precision(manifestName: "fp16") == .float16)
        #expect(Weights.Precision(manifestName: "float16") == .float16)
        #expect(Weights.Precision(manifestName: "FP32") == .float32)
        #expect(Weights.Precision(manifestName: "bf16") == .bfloat16)
        #expect(Weights.Precision(manifestName: "int8") == nil)
        for p: Weights.Precision in [.float32, .float16, .bfloat16] {
            #expect(Weights.Precision(manifestName: p.manifestName) == p)
        }
    }
}
