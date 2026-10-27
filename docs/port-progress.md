# Port progress

Hand-maintained. `test-port-manifest.md` is **generated** — don't edit it; regenerate with
`python3 -I src/rfd4_mlx_export/triage_tests.py <upstream> docs/test-port-manifest.md`.

## Upstream tests ported

**19 of 233.**

| Upstream file | Ported | Swift |
|---|---|---|
| `tests/test_atom37_encoding_alignment.py` | 5 / 5 | `Atom37EncodingAlignmentTests.swift` |
| `tests/test_recycle_depths.py` | 6 / 6 | `RecycleDepthsTests.swift` |
| `tests/test_partial_schedule.py` | 3 / 7 | `PartialScheduleTests.swift` |
| `tests/test_nucleic_ss_input.py` | 1 / 1 | `DotBracketTests.swift` |
| `tests/utils/test_structure_files.py` | 4 / 4 | `StructureFilesTests.swift` |

Swift side: 35 test functions, 90 cases, running in ~13 ms. Build is warning-clean.

## Implementation landed

| Swift | Upstream | Notes |
|---|---|---|
| `Vocabulary.swift` | `encoding.py`, `utils/constants.py`, `decoder.py:15-37` | 147 classes; `residueAtomMask` `[147][37]`, `allowedResidueTypeMask` `[7][147]` |
| `AtomArray.swift` | `biotite.structure.AtomArray`, `utils/token_encoding.py` | struct-of-arrays; `atom37Slots` with residue-qualified lookup |
| `Atom37Encoding.swift` | `atomworks.ml.transforms.encoding` | encode / decode round trip |
| `RecycleSchedule.swift` | `proteina.py:recycle_depths` | band expansion, partial tail, `trunkPasses` |
| `FlowSchedule.swift` | `product_space_flow_matcher.py`, `rdn_flow_matcher.py:553` | time + noise schedules, TSR ratio, tail multiplier, partial slicing |
| `DotBracket.swift` | `inference_utils/nucleic_ss_input.py` | dot-bracket parsing, both case conventions |
| `StructureFiles.swift` | `utils/structure_files.py` | compression-independent stems, ambiguity, sibling lookup |

## Deferred, with reasons

### From `test_partial_schedule.py` (4 tests)

- `test_schedule_starts_at_the_noised_time`
- `test_fraction_corruption_uses_configured_grid`
- `test_fraction_rejects_zero_remaining_steps`

  All three need `resolve_partial_t` (`search/vanilla.py:41`), which reads the inference
  config and the inverse-folding switches. That is config plumbing, not schedule math;
  it lands with the inference-config port.

- `test_fractional_generation_corrupts_at_first_integration_time`

  Needs `VanillaStrategy` / `BeamSearchStrategy`. Phase 2c (sampler).

### `build_partial_schedule` dense branch

Only the `full_nsteps` branch is ported. The dense branch (`full_nsteps = nil`, which builds
a `max(nsteps*100, 1000)`-point reference grid to couple explicit per-modality times) is used
only by explicit `partial_t_backbone` / `partial_t_latent`, not by fractional restarts.

## Verified against the installed upstream

The upstream package now installs on this Mac (114 packages, torch 2.14.1, **no CUDA or
Triton wheels**), so transcriptions are checked rather than inferred.

- **Vocabulary: exact.** All 147 classes, identical order, zero layout mismatches, zero
  `token_to_idx` mismatches, group sizes identical. `atom37Slots` matches
  `build_atom37_slots` for every probed residue, including the UNK-aliases-standard-AA
  rule and `-1` for an atom absent from its residue layout.

- **Structure-file suffixes: my reconstruction was WRONG, now fixed.** It missed the whole
  `.mmjson` family and every `.gzip` spelling. (`.mmjson` survives upstream's `.json`
  filter because the filter is on names that *start* with `.json`.) Pinned by a test
  against the real `STRUCTURE_FILE_SUFFIXES` so it cannot drift back.

- **torch on this Mac:** `cuda_available=False`, `mps_available=True`. The False is what
  makes `pair_bias_attn.py:173` take its explicit einsum branch here — the readable fp32
  reference the trunk port will be checked against.

## Schedule parity (Phase 3, first slice)

`FlowScheduleParityTests` compares against float32 bit patterns dumped by
`src/rfd4_mlx_export/dump_schedule_fixtures.py`. It found two real defects:

1. **`linspace` needed PyTorch's halfway split.** torch accumulates forward from `start`
   for the lower half of a grid and *backward from `end`* for the upper half. Without it
   the port sat up to **40 ULP** off through the whole second half of every schedule; with
   it, 1 ULP. The split also pins both endpoints exactly, replacing a special case.

2. **`tsr_sigma**2` is a scalar evaluated once.** Writing it as `snr * sigma * sigma`
   multiplies elementwise twice and costs 1 ULP per element. Pre-squaring makes
   `computeTSRRatio` bit-exact.

Residual, after both fixes — these are measured budgets, not guesses:

| quantity | worst | budget |
|---|---|---|
| time grids and raw builders | 3 ULP | 4 |
| `bb_ca.gt` (mode `1/t`) | 10 ULP | 192 |
| `local_latents.gt` (mode `tan`) | 125 ULP | 192 |

The 125 is **conditioning, not slop**. `tan` mode is `(pi/2)*sin(a)/(cos(a)+1e-2)`; as
`t -> 0`, `cos(a) -> 0` and the quotient turns sharply sensitive to the angle. Measured:
one ULP of `a` moves `gt` by **124 ULP** at t=0.0026, 75 at t=0.01, 2 by t=0.5. The same
formula in numpy float32 sits 2 ULP from torch, so the formula is right — what is left is
a one-ULP disagreement between Swift's `sinf`/`cosf` and torch's vectorized ones, amplified
~100x. Closing it would mean shipping our own transcendentals. At ~1.5e-5 relative it stays
an order of magnitude under the int8 floor (r >= 0.9997) the port is built for.

## Checkpoint (Phase 1 groundwork)

Weights fetched selectively — the HF repo also carries ~60 GB of AF2 / OpenFold3 / CUDA
container images that `allow_patterns` keeps out. **sha256 of the served file matches the
`weights_sha256` in `config.json`.** Inventory: `docs/checkpoint-inventory.md`
(regenerate with `src/rfd4_mlx_export/inspect_checkpoint.py`).

1257 tensors, **435,622,465 parameters** — exactly the model card's figure. Every number
the architecture map derived from configs holds against the real checkpoint:

| component | map's estimate | checkpoint | blocks |
|---|---|---|---|
| `nn.transformer_layers` | 157.1 M | 157.10 M | 14 (0–13) |
| `nn.pair_update_layers` | 17.2 M | 17.24 M | 7, at indices 0,2,4,6,8,10,12 |
| `autoencoder.encoder` | 129.0 M | 129.01 M | 12 |
| `autoencoder.decoder` | 128.1 M | 128.16 M | 12 |
| flow trunk total | 178.5 M | 178.4 M | |

Config confirms `nlayers: 14`, `token_dim: 768`, `nheads: 12`, `noise_align_chirality: True`,
`noise_align_internal_coords: False`. Stored dtype is **F32 for all 1257 tensors**, so
"full precision" is the narrowing choice the gate doc predicted, not a free upgrade.

Weight budget — **the full bundle ships, nothing dropped**. A de-novo-only port could
leave out the AE encoder, but partial diffusion, inverse folding and upstream's
`examples/antibody_design/partial.yaml` all need it, and 0.13 GB at int8 is not worth
putting an upstream example out of reach. The encoder is *stochastic*
(`encoder.py:147` reparameterises), so parity must pin mean-vs-sample explicitly.

| | params | fp32 | int8 g64 |
|---|---:|---:|---:|
| **shipped (full bundle)** | **435.6 M** | **1.74 GB** | **0.44 GB** |
| _de-novo-only subset, for reference_ | _306.6 M_ | _1.23 GB_ | _0.31 GB_ |

## The model runs on CPU — measured, not inferred

`Proteina.from_pretrained(<snapshot>, device="cpu")` loads in **1.1 s** on the M4 and the
flow trunk runs a forward pass (8 tokens, then 32 tokens in 0.08 s). cuEquivariance is
absent and falls back to torch, which is the reference path this port transcribes.

That settles the gate doc's central scheduling claim: **no CUDA box is needed for the
parity oracle.** `torch.cuda.is_available()` is False here, which is also what makes
`pair_bias_attn.py:173` take its explicit einsum branch — the readable fp32 reference.

Head shapes, all as the architecture map predicted:

| head | shape |
|---|---|
| `bb_ca.v` | [B, I, 3] |
| `local_latents.v` | [B, I, 32] |
| `sequence.logits` | [B, I, 147] |
| `insertion.logits` | [B, I, I] |

One structural detail the configs did not show: `pair_update_layers` is a `ModuleList` of
**13**, with `PairReprUpdate` at even indices and **`None` at odd ones** — so there is no
pair update after the final transformer layer. The Swift trunk must reproduce that, not
assume 7 evenly-spaced updates.

## Trunk parity fixtures (Phase 3, second slice)

`src/rfd4_mlx_export/dump_trunk_fixtures.py` captures **49 module boundaries + 4 heads** at
8 tokens into `Tests/RFD4CoreTests/Fixtures/trunk.json` (1.39 MB). Per transformer layer it
records the layer output, the `mhba` output and the `transition` output separately — all
three are distinct, so a divergence can be attributed to attention or to the MLP rather
than to "layer 7". Pair representations are stored as statistics plus a leading slice,
because [B, I, I, 384] in full would dominate the file.

Fixtures are deliberately tiny. Parity does not need large inputs, it needs exact ones;
at 64 tokens one pair capture alone would be 6 MB.

**The Swift consumer for these is not written yet** — the fixtures exist, nothing reads
them. That is the next step, and it is what Phase 2 is checked against.

## Conditioning scope, measured

Run upstream's own pipeline on the shipped `examples/protein_binder/input.cif` — a
real binder design, 225 tokens — and hook every feature creator
(`src/rfd4_mlx_export/scope_conditioned_features.py`). Result:

| regime | live creators |
|---|---|
| unconditioned de-novo | 13 of 35 |
| **binder design** | **17 of 35** |

So conditioning costs **four more features**, not the twenty-two the de-novo
measurement left open:

| newly live | width | carries |
|---|---|---|
| `ConditionalAtom37NanometersCoorsSeqFeat` | 148 | the target's actual coordinates — how the target enters the model at all |
| `SequenceProbabilitiesSeqFeat` | 56 | the target's known sequence |
| `HotspotSeqFeat` | 37 | the hotspot residues |
| `CanonicalPolymerOrderingPairFeat` | 1 | backbone linkage direction |

Eighteen stay inert even conditioned: RASA, secondary structure, hydrogen bonds,
pi-stacking, pi-cation, salt bridges, hydrophobic, metal coordination, nucleic-acid
SS (seq and pair), atom charge, atomic element, chirality, chain centroid, internal
coordinates, recycle-remaining, bond mask, is-aromatic.

**`ChiralitySeqFeat` is among them.** That retires, for this scope, the MLX spike
flagged as a possible redesign-forcer: the batched SVD in `noise_alignment.py` runs
every forward pass but early-returns when no chiral frames exist, and a conditioned
canonical-20 protein has none. The SVD only goes live with NCAAs, glycans or
ligands — the full-vocabulary scope, not binder design.

## Known gaps that are not test coverage

- **Weights have not been downloaded.** Everything above is architecture- and
  numerics-level; `Proteina.from_pretrained` (1.74 GB) has not been fetched, so no layer
  has been run and no weight-export path exists yet.

- **`transformGt` non-identity branch** is exercised only at `noisePower` 1.0 and 1.5.
  Both released configs use `p: 1.0`, the identity path.

- **Licensing unresolved.** See README. Do not publish.

## Blocked

- **Phase 0 science gate (§1b of `phase0-gate.md`) is unrun.** The RFD3-replacement regime
  is 638 tokens against a model trained at `_max_tokens: 256`. Everything landed so far is
  scope-independent foundation, but the featurizer's conditioning design should not be
  finalised before that result.

## Binder design (27 Oct 2026)

A design request now becomes a batch. `Contig` parses the shipped grammar,
`TokenLayout` derives the connectivity across a chain break, and `DesignProblem`
assembles the conditioning.

Verified against upstream's own `ConditionSpecDataset` run of
`"50-120, /0, A17-131"` on `docs/examples/inputs/5o45_cropped.cif.zst` — 225
tokens, 110 designed then 115 of the PD-L1 target:

| what | max rel |
| --- | --- |
| `token_bond_path_length` | exact |
| `canonical_polymer_ordering` | exact |
| `SequenceProbabilitiesSeqFeat` | exact |
| `ConditionalAtom37NanometersInternalCoorsSeqFeat` | exact |
| `HotspotSeqFeat`, `CoordGroupSeqFeat` | exact |
| `ConditionalAtom37NanometersCoorsSeqFeat` | exact |
| `init_repr_factory` | 2.7e-07 |
| `cond_factory` | 4.7e-07 |
| `pair_repr_builder` (16×16 corner) | 4.8e-07 |

Three things that only a conditioned run could expose:

- **Coordinate groups are not raw.** `CoordGroupSeqFeat` relabels them with a fresh
  random permutation of `0..k` on every forward pass, so for a single group the
  network always sees `0`. Passing the pipeline's own id — `1` — feeds a value no
  training example carried.
- **The sequence-probability channel order is the registry's**, not sorted: 20 amino
  acids alphabetically, then RNA, then DNA. Sorting as strings interleaves them
  (`A`, `ALA`, `ARG`, …) and cost 8% of the sequence representation. The
  per-creator bisection found it; the whole-factory number could not have.
- **The target is not frozen.** `is_in_context_conditioning` is all false on
  upstream's binder batch, so the emit masks are just the padding mask: both chains
  are generated and the target is held in place by its conditioning alone.

The free-pose recipe conditions on **internal** coordinates — the target's backbone
centred on its own centroid and randomly rotated (`s=0`, no translation). The
absolute-coordinate feature is inert there. Both are implemented.
