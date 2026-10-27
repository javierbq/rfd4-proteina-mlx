import Foundation
import MLX
import RFD4Core
import RFD4MLX

// A de-novo generation runner: produces a structure file you can open.
//
//   rfd4-generate <weights.safetensors> <out.cif> [length] [seed] [low|medium]

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write(Data("""
        usage: rfd4-generate <weights.safetensors> <out.cif> [length] [seed] [low|medium]

        Generates an unconditioned protein backbone with RFD4-Proteina running
        natively in MLX, and writes it as mmCIF.

        """.utf8))
    exit(2)
}

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
print("wrote \(outPath)")
