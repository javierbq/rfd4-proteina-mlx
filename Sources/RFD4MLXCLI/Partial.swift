import Foundation
import MLX
import RFD4Core
import RFD4MLX

/// Partial diffusion: take a structure you already have and re-run the tail of the
/// trajectory on it, so the output is a variation rather than a fresh design.
///
/// The input should be a complex this model produced, or another structure whose
/// modelled atoms match its atom37 mask. A crystal structure with unresolved
/// sidechains does not: the encoder would return an entirely NaN latent, and the
/// run refuses rather than writing a file full of NaN.
enum PartialCommand {

    struct Options {
        var weights: String
        var input: String
        var out: String
        /// How much of the trajectory to re-run. 0.3 keeps close to the input; 0.8
        /// drifts well away from it.
        var fraction = 0.4166666666666667
        var seed: UInt64 = 0
        var deterministicEncode = false
    }

    static func run(_ o: Options) throws {
        let vocab = Vocabulary.unifiedAtom37
        let raw = try AtomArray.fromMMCIF(contentsOf: URL(fileURLWithPath: o.input))
        let input = raw.normalizedForDesign()
        print("input: \(input.count) atoms from \(o.input)")

        // One token per residue, in file order, keyed by chain so two chains
        // numbered over the same range stay apart.
        struct Key: Hashable { let chain: String; let residue: Int }
        var order: [Key] = []
        var atoms: [Key: [(slot: Int, coord: Vec3, element: Int, charge: Int)]] = [:]
        let slots = input.atom37Slots(vocab)
        for i in input.indices {
            guard let rid = input.resID?[i], rid >= 0, slots[i] >= 0 else { continue }
            let key = Key(chain: input.chainID?[i] ?? "", residue: rid)
            if atoms[key] == nil { order.append(key) }
            atoms[key, default: []].append(
                (slots[i], input.coord[i], atomicNumber(input.element?[i] ?? ""), 0))
        }
        let n = order.count
        guard n > 0 else { throw CLIError.message("no residues parsed from \(o.input)") }

        var chainIndexOf: [String: Int] = [:]
        for key in order where chainIndexOf[key.chain] == nil {
            chainIndexOf[key.chain] = chainIndexOf.count
        }
        print("tokens: \(n) over \(chainIndexOf.count) chain(s)")

        // The encoder's inputs, flattened to the atom37 grid.
        let slotCount = Vocabulary.slotCount
        var coords = [Float](repeating: 0, count: n * slotCount * 3)
        var mask = [Float](repeating: 0, count: n * slotCount)
        var element = [Float](repeating: 0, count: n * slotCount)
        var charge = [Float](repeating: 0, count: n * slotCount)
        var chainIndex = [Float](repeating: 0, count: n)
        for (t, key) in order.enumerated() {
            chainIndex[t] = Float(chainIndexOf[key.chain]!)
            for atom in atoms[key]! {
                let i = t * slotCount + atom.slot
                mask[i] = 1
                element[i] = Float(atom.element)
                charge[i] = Float(atom.charge)
                // Angstrom to nanometres: the encoder works in nm, and upstream's
                // own wrapper divides by 10 before the encoder ever runs.
                coords[i * 3 + 0] = atom.coord.x / 10
                coords[i * 3 + 1] = atom.coord.y / 10
                coords[i * 3 + 2] = atom.coord.z / 10
            }
        }

        let allOnes = MLX.ones([1, n])
        let structure = Encoder.Structure(
            coordsNM: MLXArray(coords, [1, n, slotCount, 3]),
            atom37Mask: MLXArray(mask, [1, n, slotCount]),
            tokenType: MLX.zeros([1, n]),
            chainIndex: MLXArray(chainIndex, [1, n]),
            // Bond-type counts are a measurement this port does not make yet; zero
            // is what an input without them contributes, and it is stated rather
            // than hidden because it is a real difference from upstream's batch.
            incidentBondTypeCounts: MLX.zeros([1, n, slotCount, 10]),
            element: MLXArray(element, [1, n, slotCount]),
            charge: MLXArray(charge, [1, n, slotCount]),
            attnMask: allOnes, emitMask: allOnes)
        try structure.validate()

        // The layout: the chains as the file presents them.
        var tokens: [TokenLayout.Token] = []
        for key in order {
            tokens.append(TokenLayout.Token(
                chainIndex: chainIndexOf[key.chain]!, chainID: key.chain,
                segmentIndex: chainIndexOf[key.chain]!, inputResidue: key.residue))
        }
        let layout = TokenLayout(tokens: tokens)
        print("layout: breaks at \(layout.chainBreaks())")

        let problem = DesignProblem(
            layout: layout,
            coords: (0 ..< n).map { t in
                (0 ..< slotCount).map { s -> Vec3? in
                    let i = t * slotCount + s
                    guard mask[i] > 0 else { return nil }
                    return Vec3(coords[i * 3] * 10, coords[i * 3 + 1] * 10,
                                coords[i * 3 + 2] * 10)
                }
            },
            knownSequence: [Int?](repeating: nil, count: n))
        let system = try problem.validatedSystem(seed: o.seed)

        let loadStart = Date()
        let w = try Weights.loadPathOrPack(o.weights)
        let generator = try Generator(w)
        let encoder = try Encoder(w)
        print("loaded weights in \(String(format: "%.1f", Date().timeIntervalSince(loadStart)))s")

        let steps = try PartialDiffusion.remainingSteps(
            fraction: o.fraction, nsteps: ReleasedRecycleStrategy.nsteps)
        let full = try sampleSchedule(nsteps: ReleasedRecycleStrategy.nsteps,
                                      modalities: ModalityConfig.released)
        let partialT = try PartialDiffusion.partialT(
            fraction: o.fraction, nsteps: ReleasedRecycleStrategy.nsteps, schedule: full)
        print(String(format: "restart: %d of %d steps, backbone t=%.3f, sequence t=%.3f",
                     steps, ReleasedRecycleStrategy.nsteps,
                     partialT["bb_ca"] ?? 0, partialT["local_latents"] ?? 0))

        var lastReport = Date()
        let runStart = Date()
        let result = try generator.generatePartial(
            system, from: structure, encoder: encoder,
            fraction: o.fraction, seed: o.seed,
            encodeDeterministically: o.deterministicEncode
        ) { step, passes in
            if Date().timeIntervalSince(lastReport) > 10 {
                print("  step \(step + 1)/\(steps)  \(passes) passes")
                lastReport = Date()
            }
        }
        print(String(format: "generated in %.1fs", Date().timeIntervalSince(runStart)))

        try BinderCommand.writeComplex(result, layout: layout, vocab: vocab, to: o.out)

        // How far it moved: the whole point of a fraction is to control this.
        let before = MLXArray(coords, [1, n, slotCount, 3])[0, 0..., 1, 0...]
        let after = result.xFinal["bb_ca"]![0]
        let drift = MLX.sqrt(MLX.mean(MLX.sum((after - before) * (after - before), axis: -1)))
        print(String(format: "  CA drift from input: %.2f A (fraction %.3f)",
                     drift.item(Float.self) * 10, o.fraction))
        print("wrote \(o.out)")
    }

    /// Atomic number from an element symbol, for the encoder's element embedding.
    static func atomicNumber(_ symbol: String) -> Int {
        let table = ["H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "NA": 11, "MG": 12,
                     "P": 15, "S": 16, "CL": 17, "K": 19, "CA": 20, "MN": 25,
                     "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30, "SE": 34,
                     "BR": 35, "I": 53]
        return table[symbol.trimmingCharacters(in: .whitespaces).uppercased()] ?? 0
    }
}
