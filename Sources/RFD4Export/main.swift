// RFD4Export — build an installable weight pack from the upstream checkpoint.
//
//   rfd4-export <checkpoint.safetensors> <out-dir> [--precision fp16|fp32|bf16]
//                                                  [--version v1] [--no-verify]
//
// Deliberately written in Swift against the same MLX this package already depends
// on, rather than as a Python script beside the fixture dumpers. Two reasons.
//
// The cast has to be the one `Weights.load` performs — so this calls
// `Weights.narrowed`, the same function, instead of a NumPy reimplementation of
// it that could drift. And a Python exporter would need safetensors + numpy in an
// environment this repo does not pin, which is how an export step becomes
// unreproducible six months later.
//
// The pack it writes is flat: `manifest.json` + `rfd4_core.safetensors`, no
// subdirectories, because RayMol's weight cache rejects any zip member that is not
// a plain file name.
import Foundation
import MLX
import RFD4Core
import RFD4MLX

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("error: " + message + "\n").utf8))
    exit(1)
}

func usage() -> Never {
    print("""
    usage:
      rfd4-export <checkpoint.safetensors> <out-dir> [options]

    options:
      --precision fp16|fp32|bf16   weight dtype to store      (default fp16)
      --version <string>           pack version               (default v1)
      --name <string>              model name in the manifest (default RFD4-Proteina)
      --no-verify                  skip the read-back check   (not recommended)

    Writes <out-dir>/manifest.json and <out-dir>/rfd4_core.safetensors.
    """)
    exit(2)
}

// MARK: - arguments

var args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 2, !args[0].hasPrefix("--"), !args[1].hasPrefix("--") else { usage() }
let checkpointPath = args.removeFirst()
let outPath = args.removeFirst()

var precision: Weights.Precision = .float16
var packVersion = "v1"
var modelName = "RFD4-Proteina"
var verify = true

var i = 0
while i < args.count {
    switch args[i] {
    case "--precision":
        i += 1
        guard i < args.count, let p = Weights.Precision(manifestName: args[i]) else {
            fail("--precision takes fp16, fp32 or bf16")
        }
        precision = p
    case "--version":
        i += 1
        guard i < args.count else { usage() }
        packVersion = args[i]
    case "--name":
        i += 1
        guard i < args.count else { usage() }
        modelName = args[i]
    case "--no-verify":
        verify = false
    default:
        fail("unknown option \(args[i])")
    }
    i += 1
}

let checkpointURL = URL(fileURLWithPath: (checkpointPath as NSString).expandingTildeInPath)
let outURL = URL(fileURLWithPath: (outPath as NSString).expandingTildeInPath)
guard FileManager.default.fileExists(atPath: checkpointURL.path) else {
    fail("no checkpoint at \(checkpointURL.path)")
}

// MARK: - read, narrow, write

print("reading   \(checkpointURL.path)")
// The checkpoint path is routinely a symlink into the HuggingFace blob store, so
// hash the resolved file but NAME the link. The blob is called something like
// `092c821aa79f...`, which as a provenance field tells a reader nothing and looks
// like a hash that disagrees with the sha256 beside it; `model.safetensors` plus
// that sha256 is what identifies the thing.
let resolvedCheckpoint = checkpointURL.resolvingSymlinksInPath()
let sourceName = checkpointURL.lastPathComponent
let sourceSHA = try RFD4Pack.sha256Hex(of: resolvedCheckpoint)
let raw = try MLX.loadArrays(url: checkpointURL)

let sourceDTypes = Set(raw.values.map { "\($0.dtype)" }).sorted()
let parameterCount = raw.values.reduce(0) { $0 + $1.size }
print("          \(raw.count) tensors, \(parameterCount) parameters, dtypes \(sourceDTypes)")
print("          sha256 \(sourceSHA)")

print("narrowing to \(precision.manifestName)")
let narrowed = Weights.narrowed(raw, to: precision)
// Force the casts before writing: MLX is lazy, so without this the elapsed time and
// any failure would land inside `save` rather than here.
MLX.eval(Array(narrowed.values))

try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)
let weightsURL = outURL.appendingPathComponent(RFD4Pack.weightsFilename)

let vocabulary = Vocabulary.unifiedAtom37
let exportedAt = ISO8601DateFormatter().string(from: Date())

// Stamped into the safetensors header as well as the manifest. The manifest can be
// edited by hand or lost in a repack; the header travels with the bytes, so a
// stray weights file is still identifiable.
let metadata: [String: String] = [
    "format": "pt",
    "rfd4_pack_format": RFD4Pack.supportedFormat,
    "rfd4_precision": precision.manifestName,
    "rfd4_source_checkpoint": sourceName,
    "rfd4_source_sha256": sourceSHA,
    "rfd4_vocabulary": vocabulary.name,
    "rfd4_vocabulary_ccd": vocabulary.ccdSnapshot,
    "rfd4_exported_at": exportedAt,
]

print("writing   \(weightsURL.path)")
try MLX.save(arrays: narrowed, metadata: metadata, url: weightsURL)

let weightsSHA = try RFD4Pack.sha256Hex(of: weightsURL)
let weightsBytes = (try FileManager.default.attributesOfItem(atPath: weightsURL.path)[.size]
    as? NSNumber)?.intValue ?? 0
print("          \(weightsBytes) bytes, sha256 \(weightsSHA)")

let manifest = RFD4Pack.Manifest(
    name: modelName,
    version: packVersion,
    format: RFD4Pack.supportedFormat,
    precision: precision.manifestName,
    weights: RFD4Pack.Provenance(
        sourceCheckpoint: sourceName,
        sourceCheckpointSHA256: sourceSHA,
        exportedAt: exportedAt,
        exportedBy: "rfd4-export (rfd4-proteina-mlx)"),
    vocabulary: RFD4Pack.VocabularyPin(
        name: vocabulary.name,
        ccdSnapshot: vocabulary.ccdSnapshot,
        classCount: vocabulary.tokenCount),
    files: [RFD4Pack.weightsFilename: weightsSHA],
    quantization: nil)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let manifestURL = outURL.appendingPathComponent(RFD4Pack.manifestFilename)
try encoder.encode(manifest).write(to: manifestURL)
print("writing   \(manifestURL.path)")

// MARK: - read back

// An exporter that cannot prove its own output loads is just a file writer. This
// re-opens the pack through the exact path the app will use, checksums included.
if verify {
    print("verifying (loading the pack back through RFD4Pack)")
    let pack = try RFD4Pack(directory: outURL)
    guard pack.weights.arrays.count == raw.count else {
        fail("pack has \(pack.weights.arrays.count) tensors, checkpoint had \(raw.count)")
    }
    var mismatched: [String] = []
    for (key, original) in raw {
        guard let stored = pack.weights.arrays[key] else {
            mismatched.append("\(key): absent")
            continue
        }
        if stored.shape != original.shape {
            mismatched.append("\(key): shape \(stored.shape) != \(original.shape)")
        }
        let want = original.dtype == .float32 ? precision.dtype : original.dtype
        if stored.dtype != want {
            mismatched.append("\(key): dtype \(stored.dtype) != \(want)")
        }
    }
    guard mismatched.isEmpty else {
        fail("read-back mismatch:\n  " + mismatched.prefix(10).joined(separator: "\n  "))
    }
    print("          \(pack.weights.arrays.count) tensors, shapes and dtypes as expected")
}

let zipName = "rfd4-proteina-mlx-\(precision.manifestName)-\(packVersion).zip"
print("""

pack written to \(outURL.path)

  Zip it FLAT -- RayMol's weight cache compares the member set against the
  bundle's declared `members` and rejects anything that is not a plain file name:

    cd \(outURL.path) && zip -X ../\(zipName) \
    \(RFD4Pack.manifestFilename) \(RFD4Pack.weightsFilename)
    shasum -a 256 ../\(zipName) && wc -c < ../\(zipName)

  Then hash the asset you actually upload, not this local build.
""")
