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

    /// Load from either a pack directory or a bare `.safetensors` file.
    public init(weightsAtPathOrPack path: String) throws {
        try self.init(Weights.loadPathOrPack(path))
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
        /// What a conditioned design supplies beyond the layout; `nil` for de novo.
        public var conditioning: Featurizer.Batch.Conditioning?

        public init(
            tokenCount: Int, tokenType: MLXArray, paddingMask: MLXArray,
            cathCode: MLXArray, coordGroup: MLXArray, indexGroup: MLXArray,
            moleculeIID: MLXArray, moleculeEntityID: MLXArray,
            bondOrder: MLXArray, tokenBondPathLength: MLXArray,
            canonicalPolymerOrdering: MLXArray,
            emitBBCA: MLXArray, emitLocalLatents: MLXArray,
            conditioning: Featurizer.Batch.Conditioning? = nil
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
            self.conditioning = conditioning
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
            canonicalPolymerOrdering: s.canonicalPolymerOrdering,
            conditioning: s.conditioning)
    }

    /// Restart a trajectory from a structure instead of from noise.
    ///
    /// Corrupt the input to a point on the configured schedule, then denoise only
    /// the tail. The fraction names how many of the full trajectory's steps to run,
    /// and the corruption time is read off the schedule at that index — see
    /// `PartialDiffusion`, and note the two modalities land at very different times
    /// from one fraction.
    ///
    /// - Parameters:
    ///   - structure: what to start from. Its latents come from the encoder, which
    ///     is why this takes one; `bb_ca` is the CA row of its atom37 coordinates,
    ///     in nanometres and uncentred, which is what upstream's `x_1` holds.
    ///   - fraction: how much of the trajectory to re-run. Smaller keeps more of the
    ///     input.
    ///   - encodeDeterministically: take the encoder's mean rather than a draw. The
    ///     released flow samples, so this is off by default; it exists because a
    ///     caller comparing two partial runs wants the latent fixed.
    public func generatePartial(
        _ system: System,
        from structure: Encoder.Structure,
        encoder: Encoder,
        fraction: Double,
        seed: UInt64 = 0,
        encodeDeterministically: Bool = false,
        stochastic: Bool = true,
        onStep: ((Int, Int) -> Void)? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) throws -> Result {
        // An all-NaN latent is not a degraded answer; refuse before spending the run.
        try structure.validate()

        let fullSteps = ReleasedRecycleStrategy.nsteps
        let full = try sampleSchedule(nsteps: fullSteps, modalities: ModalityConfig.released)
        let nsteps = try PartialDiffusion.remainingSteps(fraction: fraction,
                                                         nsteps: fullSteps)
        let partialT = try PartialDiffusion.partialT(fraction: fraction,
                                                     nsteps: fullSteps, schedule: full)
        let tail = try PartialDiffusion.tail(of: full, nsteps: nsteps)

        var key = MLXRandom.key(seed)
        func draw(_ shape: [Int]) -> MLXArray {
            let keys = MLXRandom.split(key: key, into: 2)
            key = keys[0]
            return MLXRandom.normal(shape, key: keys[1])
        }

        let latents = encoder(structure, seed: encodeDeterministically ? nil : seed)
        let n = system.tokenCount
        let x1: [String: MLXArray] = [
            // Slot 1 is CA. Uncentred: upstream's x_1 is exactly coords/10.
            "bb_ca": structure.coordsNM[0..., 0..., 1, 0...],
            "local_latents": latents.z,
        ]
        let emitMasks = ["bb_ca": system.emitBBCA, "local_latents": system.emitLocalLatents]

        let corrupted = Corruption.corrupt(
            x1: x1, t: partialT, emitMasks: emitMasks,
            groupIDs: system.moleculeIID,
            baseNoise: { _, shape in draw(shape) },
            offsetDraw: { _, _, dim in draw([1, dim]) })

        var pass = 0
        let depths = Array(repeating: 1, count: nsteps)
        let xFinal = try Sampler.simulate(
            x0: corrupted.xt, schedule: tail, nsteps: nsteps, recycleDepths: depths,
            params: Sampler.StepParams.released, emitMasks: emitMasks,
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
            onStep: { step, _ in onStep?(step, pass) },
            shouldCancel: shouldCancel)

        let decoded = decoder(
            seqInit: decoder.sequenceFeatures(
                latents: xFinal["local_latents"]!, caCoordsNM: xFinal["bb_ca"]!,
                tokenType: system.tokenType),
            cond: MLX.zeros([1, n, 128]),
            attnMask: system.emitBBCA,
            emitMask: system.emitLocalLatents,
            caCoordsNM: xFinal["bb_ca"]!,
            tokenType: system.tokenType.asType(.int32))

        return Result(coordsNM: decoded.coordsNM, aatype: decoded.aatype,
                      atomMask: decoded.atomMask, xFinal: xFinal)
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
        onStep: ((Int, Int) -> Void)? = nil,
        shouldCancel: (() -> Bool)? = nil
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
        let xFinal = try Sampler.simulate(
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
            },
            shouldCancel: shouldCancel
        )

        // Feed the decoder what the flow model actually produced. Passing zeros
        // here yields a constant sequence — every residue the same class — which
        // looks like a plausible structure file and is not one.
        //
        // The conditioning really is zeros: the decoder's cond_factory carries no
        // parameters, so it is a zero feature by construction.
        // `decoder.py:186` takes BOTH its masks from `compute_ae_masks`, which are
        // the same two expressions the flow emit masks use:
        //
        //     attn_mask = padding & ~is_in_context_conditioning   == emitBBCA
        //     emit_mask = attn_mask & supports_local_latent       == emitLocalLatents
        //
        // Passing `paddingMask` and `emitBBCA` here instead is identical for every
        // run this port can currently produce — all three are all-ones for a de-novo
        // monomer and for a binder — and one mask too permissive at each position
        // the moment partial diffusion sets context or an atomized token clears
        // supports_local_latent.
        let decoded = decoder(
            seqInit: decoder.sequenceFeatures(
                latents: xFinal["local_latents"]!,
                caCoordsNM: xFinal["bb_ca"]!,
                tokenType: system.tokenType),
            cond: MLX.zeros([1, n, 128]),
            attnMask: system.emitBBCA,
            emitMask: system.emitLocalLatents,
            caCoordsNM: xFinal["bb_ca"]!,
            tokenType: system.tokenType.asType(.int32))

        return Result(coordsNM: decoded.coordsNM, aatype: decoded.aatype,
                      atomMask: decoded.atomMask, xFinal: xFinal)
    }
}
