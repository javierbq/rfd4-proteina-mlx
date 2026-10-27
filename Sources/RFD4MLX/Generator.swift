import Foundation
import MLX
import MLXRandom
import RFD4Core

/// End-to-end generation: featurize, run the trunk, integrate, decode.
///
/// Ties the four verified pieces together. Per sampling step it rebuilds all three
/// trunk inputs from the current noised state — which is required, not an
/// optimisation opportunity: `s_init` and `z_init` both depend on `x_t`, and
/// `z_init` and `cond` both depend on `t`.
public struct Generator {
    public let featurizer: Featurizer
    public let trunk: Trunk
    public let decoder: Decoder

    public init(_ w: Weights) throws {
        featurizer = try Featurizer(w)
        trunk = try Trunk(w)
        decoder = try Decoder(w)
    }

    public init(weightsAt url: URL) throws {
        try self.init(Weights.load(url))
    }

    /// Everything about the system that does not change between sampling steps.
    ///
    /// For a single-chain de-novo protein these are all constants, but they are
    /// taken as values rather than synthesised so the same type serves a
    /// conditioned run, where they carry the target.
    public struct System {
        public var tokenCount: Int
        public var tokenType: MLXArray
        public var paddingMask: MLXArray
        public var cathCode: MLXArray
        public var coordGroup: MLXArray
        public var indexGroup: MLXArray
        public var moleculeIID: MLXArray
        public var moleculeEntityID: MLXArray
        public var bondOrder: MLXArray
        public var tokenBondPathLength: MLXArray
        /// `+1` at [downstream, upstream], `-1` transposed, 0 elsewhere.
        public var canonicalPolymerOrdering: MLXArray
        public var emitBBCA: MLXArray
        public var emitLocalLatents: MLXArray

        public init(
            tokenCount: Int, tokenType: MLXArray, paddingMask: MLXArray,
            cathCode: MLXArray, coordGroup: MLXArray, indexGroup: MLXArray,
            moleculeIID: MLXArray, moleculeEntityID: MLXArray,
            bondOrder: MLXArray, tokenBondPathLength: MLXArray,
            canonicalPolymerOrdering: MLXArray,
            emitBBCA: MLXArray, emitLocalLatents: MLXArray
        ) {
            self.tokenCount = tokenCount
            self.tokenType = tokenType
            self.paddingMask = paddingMask
            self.cathCode = cathCode
            self.coordGroup = coordGroup
            self.indexGroup = indexGroup
            self.moleculeIID = moleculeIID
            self.moleculeEntityID = moleculeEntityID
            self.bondOrder = bondOrder
            self.tokenBondPathLength = tokenBondPathLength
            self.canonicalPolymerOrdering = canonicalPolymerOrdering
            self.emitBBCA = emitBBCA
            self.emitLocalLatents = emitLocalLatents
        }

        /// An unconditioned single-chain protein of `length` residues.
        ///
        /// **These fields must describe a POLYMER, not just n residues.** The
        /// connectivity ones are what tell the network the tokens form one chain in
        /// order; filling them with the schema's defaults instead says the opposite,
        /// and the model obeys: it generates a cloud of unbonded fragments with no
        /// backbone. Values here are read off a real de-novo request built through
        /// upstream's own dataset and pipeline, not from the schema defaults.
        ///
        ///   - `tokenBondPathLength` = `min(|i-j|, pathLengthCap)`. Upstream clamps
        ///     to the BFS cutoff plus one; all-zeros means "every pair disconnected".
        ///   - `canonicalPolymerOrdering` = `+1` at `[i+1, i]` and `-1` at `[i, i+1]`,
        ///     the direction of each backbone linkage. All-zeros means "no backbone".
        ///   - `tokIdx` runs 0..<n, and the chain, molecule, entity and index-group
        ///     ids are **0**, not -1: -1 reads as "ungrouped", so every token looks
        ///     like it belongs to a different molecule.
        public static func deNovo(length n: Int, pathLengthCap: Int = 65) -> System {
            func filled(_ v: Float, _ shape: [Int]) -> MLXArray {
                MLX.full(shape, values: MLXArray(v))
            }
            let idx = MLXArray(Array(0 ..< n).map(Float.init))

            // |i - j|, clamped.
            let diff = MLX.abs(idx.expandedDimensions(axis: 1) - idx.expandedDimensions(axis: 0))
            let pathLength = MLX.minimum(diff, Float(pathLengthCap)).expandedDimensions(axis: 0)

            // +1 at [downstream, upstream]; -1 at the transpose.
            let delta = idx.expandedDimensions(axis: 1) - idx.expandedDimensions(axis: 0)
            let adjacent = MLX.abs(delta) .== 1
            let ordering = MLX.where(adjacent, MLX.sign(delta), MLXArray(Float(0)))
                .expandedDimensions(axis: 0)

            return System(
                tokenCount: n,
                tokenType: filled(0, [1, n]),
                paddingMask: filled(1, [1, n]),
                cathCode: filled(-1, [1, n, 3]),
                coordGroup: filled(-1, [1, n, 37]),
                indexGroup: filled(0, [1, n]),
                moleculeIID: filled(0, [1, n]),
                moleculeEntityID: filled(1, [1, n]),
                bondOrder: filled(0, [1, n, n]),
                tokenBondPathLength: pathLength,
                canonicalPolymerOrdering: ordering,
                emitBBCA: filled(1, [1, n]),
                emitLocalLatents: filled(1, [1, n]))
        }
    }

    public struct Result {
        /// `[b, n, 37, 3]` all-atom coordinates in nanometres.
        public let coordsNM: MLXArray
        /// `[b, n]` residue class per token.
        public let aatype: MLXArray
        /// `[b, n, 37]` which slots carry an atom.
        public let atomMask: MLXArray
        /// Final integrated state, for inspection.
        public let xFinal: [String: MLXArray]
    }

    private func batch(_ s: System, x: [String: MLXArray], t: [String: Float])
        -> Featurizer.Batch {
        Featurizer.Batch(
            xtBBCA: x["bb_ca"]!, xtLocalLatents: x["local_latents"]!,
            tBBCA: t["bb_ca"]!, tLocalLatents: t["local_latents"]!,
            tokenType: s.tokenType, paddingMask: s.paddingMask,
            cathCode: s.cathCode, coordGroup: s.coordGroup,
            indexGroup: s.indexGroup, moleculeIID: s.moleculeIID,
            moleculeEntityID: s.moleculeEntityID, bondOrder: s.bondOrder,
            tokenBondPathLength: s.tokenBondPathLength,
            canonicalPolymerOrdering: s.canonicalPolymerOrdering)
    }

    /// Run the full trajectory and decode the result.
    ///
    /// - Parameters:
    ///   - strategy: which recycle schedule to use. `thinkingMedium` is the released
    ///     default and costs 180 trunk passes over 60 sampling steps.
    ///   - seed: pins both the initial noise and the per-step injection, so two runs
    ///     with the same seed are the same sample. Without that, two "identical"
    ///     generations are two different draws and nothing can be attributed.
    public func generate(
        _ system: System,
        strategy: ReleasedRecycleStrategy = .thinkingMedium,
        seed: UInt64 = 0,
        stochastic: Bool = true,
        recycleDepths depthOverride: [Int]? = nil,
        onStep: ((Int, Int) -> Void)? = nil
    ) throws -> Result {
        let nsteps = ReleasedRecycleStrategy.nsteps
        let schedule = try sampleSchedule(nsteps: nsteps, modalities: ModalityConfig.released)
        // An override exists so a run can be matched to a reference trace;
        // comparing two runs at different depths is not a comparison.
        let depths = try depthOverride
            ?? strategy.config.depths(nsteps: nsteps, fullNsteps: nsteps)

        // A LOCAL key, not MLXRandom.seed(). Seeding is process-global state, so
        // any other draw anywhere in the process — a test running in parallel, a
        // second generation — interleaves with this trajectory and makes a seeded
        // run unreproducible. That is not a test-only concern: the playbook's
        // matched-noise accuracy work compares two runs that must differ ONLY in
        // precision, which a shared generator quietly breaks.
        var key = MLXRandom.key(seed)
        func draw(_ shape: [Int]) -> MLXArray {
            let keys = MLXRandom.split(key: key, into: 2)
            key = keys[0]
            return MLXRandom.normal(shape, key: keys[1])
        }

        let n = system.tokenCount
        let x0: [String: MLXArray] = [
            "bb_ca": draw([1, n, 3]),
            "local_latents": draw([1, n, 32]),
        ]

        var pass = 0
        let xFinal = Sampler.simulate(
            x0: x0, schedule: schedule, nsteps: nsteps, recycleDepths: depths,
            params: Sampler.StepParams.released,
            emitMasks: ["bb_ca": system.emitBBCA, "local_latents": system.emitLocalLatents],
            predict: { x, t, depth in
                let b = self.batch(system, x: x, t: t)
                let (sInit, zInit, cond) = self.featurizer(b)
                let out = self.trunk(
                    sInit: sInit, zInit: zInit, cond: cond, mask: system.paddingMask,
                    emitBBCA: system.emitBBCA, emitLocalLatents: system.emitLocalLatents,
                    nRecycle: depth)
                pass += max(1, depth)
                return ["bb_ca": out.bbCA, "local_latents": out.localLatents]
            },
            noiseFor: { _, _, shape in stochastic ? draw(shape) : nil },
            onStep: { step, x in
                if ProcessInfo.processInfo.environment["RFD4_TRACE"] != nil,
                   step % 10 == 0 || step == nsteps - 1 {
                    let ca = x["bb_ca"]![0]                       // [n, 3]
                    let n = ca.dim(0)
                    let d = ca[1...] - ca[..<(n - 1)]
                    let consec = MLX.sqrt(MLX.sum(d * d, axis: -1))
                    let centred = ca - ca.mean(axis: 0, keepDims: true)
                    let rg = MLX.sqrt(MLX.mean(MLX.sum(centred * centred, axis: -1)))
                    print(String(format: "  step %2d  Rg %.3f nm   CA-CA mean %.3f nm "
                                 + "(min %.3f max %.3f)", step, rg.item(Float.self),
                                 consec.mean().item(Float.self),
                                 consec.min().item(Float.self),
                                 consec.max().item(Float.self)))
                }
                onStep?(step, pass)
            }
        )

        // Feed the decoder what the flow model actually produced. Passing zeros
        // here yields a constant sequence — every residue the same class — which
        // looks like a plausible structure file and is not one.
        //
        // The conditioning really is zeros: the decoder's cond_factory carries no
        // parameters, so it is a zero feature by construction.
        let decoded = decoder(
            seqInit: decoder.sequenceFeatures(
                latents: xFinal["local_latents"]!,
                caCoordsNM: xFinal["bb_ca"]!,
                tokenType: system.tokenType),
            cond: MLX.zeros([1, n, 128]),
            attnMask: system.paddingMask,
            emitMask: system.emitBBCA,
            caCoordsNM: xFinal["bb_ca"]!,
            tokenType: system.tokenType.asType(.int32))

        return Result(coordsNM: decoded.coordsNM, aatype: decoded.aatype,
                      atomMask: decoded.atomMask, xFinal: xFinal)
    }
}
