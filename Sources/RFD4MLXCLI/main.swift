import Foundation
import MLX
import RFD4Core
import RFD4MLX

// A de-novo generation runner: produces a structure file you can open.
//
//   rfd4-generate <weights.safetensors> <out.cif> [length] [seed] [low|medium]

let args = CommandLine.arguments

func usage() -> Never {
    FileHandle.standardError.write(Data("""
        usage:
          rfd4-generate <weights> <out.cif> [length] [seed] [low|medium]
          rfd4-generate binder <weights> <target.cif> <out.cif> [length] [seed] [low|medium]
          rfd4-generate partial <weights> <input.cif> <out.cif> [fraction] [seed]

        Runs RFD4-Proteina natively in MLX and writes mmCIF.

        The first form generates an unconditioned backbone. The third re-runs the
        tail of the trajectory on a structure you already have, so the output is a
        variation of it rather than a fresh design; a smaller fraction stays closer
        to the input.

        The second designs a binder against a target, using the shipped recipe:
        contig "50-120, /0, A17-131", the target's backbone as one rigid group with
        a free pose, its sequence given, and hotspots on A56, A115 and A123.

        """.utf8))
    exit(2)
}

if args.count > 1, args[1] == "partial" {
    guard args.count >= 5 else { usage() }
    var options = PartialCommand.Options(weights: args[2], input: args[3], out: args[4])
    if args.count > 5, let f = Double(args[5]) { options.fraction = f }
    if args.count > 6 { options.seed = UInt64(args[6]) ?? 0 }
    do {
        try PartialCommand.run(options)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

if args.count > 1, args[1] == "binder" {
    guard args.count >= 5 else { usage() }
    var options = BinderCommand.Options(weights: args[2], target: args[3], out: args[4])
    if args.count > 5 { options.designLength = Int(args[5]) }
    if args.count > 6 { options.seed = UInt64(args[6]) ?? 0 }
    if args.count > 7, args[7] == "medium" { options.strategy = .thinkingMedium }
    do {
        try BinderCommand.run(options)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

guard args.count >= 3 else { usage() }

let weightsPath = (args[1] as NSString).expandingTildeInPath
let outPath = args[2]
let length = args.count > 3 ? Int(args[3]) ?? 60 : 60
let seed = args.count > 4 ? UInt64(args[4]) ?? 0 : 0
let strategy: ReleasedRecycleStrategy = (args.count > 5 && args[5] == "medium")
    ? .thinkingMedium : .thinkingLow

let started = Date()
let generator = try Generator(weightsAt: URL(fileURLWithPath: weightsPath))
print("loaded weights in \(String(format: "%.1f", Date().timeIntervalSince(started)))s")

let system = Generator.System.deNovo(length: length)
let runStart = Date()
var lastReport = Date()

let result = try generator.generate(system, strategy: strategy, seed: seed) { step, passes in
    if Date().timeIntervalSince(lastReport) > 5 {
        let pct = Int(Double(step + 1) / Double(ReleasedRecycleStrategy.nsteps) * 100)
        print("  step \(step + 1)/\(ReleasedRecycleStrategy.nsteps)  "
              + "\(passes) trunk passes  \(pct)%")
        lastReport = Date()
    }
}
let elapsed = Date().timeIntervalSince(runStart)

// Flatten the atom37 output to the atoms the decoder said are present.
let vocab = Vocabulary.unifiedAtom37
let coords = result.coordsNM
let mask = result.atomMask
let aatype = result.aatype

var resNames: [String] = [], atomNames: [String] = [], xyz: [Vec3] = []
var resIDs: [Int] = [], elements: [String] = []
for i in 0 ..< length {
    let cls = Int(aatype[0, i].item(Int32.self))
    guard vocab.classes.indices.contains(cls) else { continue }
    let residue = vocab.classes[cls]
    for slot in 0 ..< Vocabulary.slotCount where mask[0, i, slot].item(Bool.self) {
        let atom = residue.atoms[slot].trimmingCharacters(in: .whitespaces)
        guard !atom.isEmpty else { continue }
        resNames.append(residue.name)
        atomNames.append(atom)
        elements.append(String(atom.prefix(1)))
        resIDs.append(i + 1)
        xyz.append(Vec3(coords[0, i, slot, 0].item(Float.self),
                        coords[0, i, slot, 1].item(Float.self),
                        coords[0, i, slot, 2].item(Float.self)))
    }
}

var array = AtomArray(resName: resNames, atomName: atomNames, coord: xyz)
array.resID = resIDs
array.element = elements
array.chainID = [String](repeating: "A", count: resNames.count)

try CIFWriter.write(array, to: URL(fileURLWithPath: outPath), name: "rfd4_denovo")

// CA geometry, as a sanity line rather than a quality claim.
let ca = coords[0, 0..., 1, 0...]
let d = ca[1...] - ca[..<(length - 1)]
let caca = MLX.sqrt(MLX.sum(d * d, axis: -1)) * 10
print(String(format: "generated %d residues, %d atoms in %.1fs (%@)",
             length, resNames.count, elapsed, strategy.rawValue))
print(String(format: "  CA-CA %.2f A (min %.2f max %.2f)",
             caca.mean().item(Float.self), caca.min().item(Float.self),
             caca.max().item(Float.self)))
// MLX's own counter, not RSS. On unified memory the weights are mmap'd and the
// working set lives in Metal buffers, so `maximum resident set size` reports a
// fraction of the truth — 0.71 GB for a model whose fp32 parameters alone are
// 1.74 GB.
print(String(format: "  peak GPU memory %.2f GB (MLX), active %.2f GB",
             Double(MLX.GPU.peakMemory) / 1_073_741_824,
             Double(MLX.GPU.activeMemory) / 1_073_741_824))
print("wrote \(outPath)")
