# Phase 0 gate — `rfd4-proteina-mlx`

**Scope decided:** full replacement of the shipping RFD3 backend — `binder_design(runtime='rfd4')`
at parity with the RFD3 path (target-conditioned, hotspots, canonical-20 protein).

**Date:** 2026-10-26
**Upstream:** `RosettaCommons/RFD4-Proteina`, BSD-3-Clause, weights `nvidia/RFD4-Proteina` (HF).
**Status:** **CONDITIONAL GO** — one blocking experiment (§1) must run before any Swift.

---

## 0. What RFD4-Proteina is

Flow-matching (not diffusion) joint sequence + all-atom generator, paired with a partial
autoencoder. Released 19–21 Oct 2026 by IPD + NVIDIA.

The architectural shape that matters: the **flow trunk operates per-token on `(CA_xyz, z₃₂)`**,
and all-atom geometry is produced by the **AE decoder** at the end. This is a different cost
profile from RFD3's atom-level EDM diffusion — see §2.

| | RFD3 (shipping) | RFD4-Proteina |
|---|---|---|
| Formulation | all-atom EDM diffusion | product-space flow matching, per-token latent |
| Trunk work | 200 steps × 2 recycles | 60 steps → **180 trunk passes** (`thinking_medium`) |
| Pair track | — | pair rep `[n,n,384]`, 7 × triangle-mult (O(n³)) |
| Inference params | — | ~307 M (178 M flow + 128 M AE decoder) |
| Equivariance | — | **none** — non-equivariant trunk, geometry via noised CA + distance bins |

---

## 1. Science gate — **the blocking item**

### 1a. The metric question is inherited, not new

RFD4 has **no confidence head**, same as RFD3. RayMol's existing answer applies unchanged:
generate with RFD4 → refold the pair with on-device Boltz-2/Protenix → gate on `min_ipSAE`.
The funnel already exists. This is *not* a new science risk.

### 1b. The real gate: **token-length generalization**

`_max_tokens: 256` — the model was trained at 256 tokens. The RFD3 replacement regime is
**638 tokens** (578-residue HSA target + 60-residue design), i.e. **2.5× beyond training
length**. RFD3 was trained and validated in that regime; RFD4 was not.

> **Nothing in the model card, the repo, or the published benchmarks establishes that
> RFD4-Proteina conditions correctly on a 578-residue target.** The headline result
> (158 binders from 188 designs) does not state target sizes.

**This is the experiment that gates the project**, and it is cheap:

1. Run upstream RFD4 **on CPU on the M4** (see §3 — it works) against the same HSA target +
   hotspots the RFD3 port was benchmarked on.
2. Production settings: `thinking_medium` (180 passes), not a fast preset.
3. Refold with Boltz-2, compute `min_ipSAE`, and compare head-to-head against the RFD3
   designs you already have.
4. Check target drift — RFD3's contract check reads 0.000 Å; RFD4 must hold the target fixed too.

**Fail condition:** if RFD4 does not condition correctly at 638 tokens, the full-replacement
scope is dead and the project reverts to the narrow de-novo slice (which is still valuable, but
is a *different* product decision and must go back to you).

Per the skill: *"Do not skip because the paper says it works — the paper's regime is rarely
RayMol's."* This is exactly that case.

---

## 2. Memory & time model

**These are desk estimates from layer shapes, not measurements.** The skill requires a
conservative fit against measured points before any run is licensed; that fit is Phase 5 work.
Nothing here authorizes removing a size guard.

### FLOPs (n = 638 tokens, 180 trunk passes)

Triangle multiplication dominates — it is the only O(n³) op.

| Term | Per trunk pass | × 180 passes |
|---|---|---|
| Triangle mult (2 × 7 layers, `ikc,jkc→ijc`, c=256) | 1.86e12 | **3.35e14** |
| Linear/SwiGLU (14 layers, ~130 M params) | 1.7e11 | 3.0e13 |
| Pair-bias attention (14 layers) | 1.75e10 | 3.1e12 |
| **Total** | | **≈ 3.7e14 FLOPs** |

At a sustained 2–4 TFLOP/s on M4: **~90–185 s per design.**

> **If that holds, RFD4 is ~6–10× faster than RFD3's measured 17 min at the same token count** —
> consistent in direction with the "5–10× faster" claim, though that claim was made on A100s.
> Treat as a hypothesis to measure in Phase 5, not a number to quote.

### Peak memory (n = 638, bf16)

| Tensor | Size |
|---|---|
| Pair rep `[n,n,384]` (persistent across all recycles) | 313 MB |
| Triangle-mult intermediates `[256,n,n]` × 4 (grouped GEMM) | ~830 MB transient |
| Attention bias `[12,n,n]` | 10 MB |
| Weights (int8 g64, flow + AE decoder) | ~310 MB |

**Estimated peak ≈ 1.5–2.5 GB at n=638.** Against RFD3's ~18 GB on a 36 GiB Mac this is a large
improvement, and it is the direct consequence of the trunk being per-token rather than per-atom.
Plausibly within iPhone reach at n≤256 — but RFD3MLX is macOS-only for good reasons and that
filter should stay until measured.

**Unchanged hazard:** MLX OOM from a Metal completion handler is `std::terminate` — uncatchable.
The port needs an `RFD4Budget` pre-flight refusal mirroring `RFD3Budget`, fitted conservatively
and with a test asserting the fit never falls under a measured point.

---

## 3. Dependency & oracle resolution

### The parity oracle runs on your Mac — no CUDA box needed

This is the most valuable finding for scheduling. Upstream docs say *"Linux x86_64, NVIDIA GPU,
macOS and Windows are not supported"*, and `generate.py:709` hardcodes `accelerator="gpu"`.
**But both networks are device-agnostic PyTorch.** Bypassing `generate.py` and calling
`Proteina.from_pretrained(...)` + `fm.full_simulation(...)` directly runs on CPU.

Better still, `nn/modules/pair_bias_attn.py:173` branches on **global** `torch.cuda.is_available()`,
not tensor device — so on a Mac it always takes an explicit, readable einsum softmax path:

```python
if not torch.cuda.is_available():
    sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
    if attn_bias is not None: sim = sim + attn_bias
    attn = torch.softmax(sim, dim=-1)
    o = einsum("b h i j, b h j d -> b h i d", attn, v)
```

**That is your fp32 ground truth for the attention port**, already unfused. Verified by reading.

### Python deps — resolvable on macOS, not yet proven

- `atomworks` 3.0.4 and `anarcii` 2.0.8 are **pure-Python wheels** (`py3-none-any`).
- `openbabel-wheel` ships **macOS arm64 wheels** — and is in any case only pulled in by the
  *metrics* path, not generation.
- CUDA-13 torch index is only needed for the GPU path; CPU torch on darwin-arm64 is fine.

**Not proven:** I could not run a resolution. `uv pip compile` was blocked by the sandbox because
it would build external packages from source. **This needs your approval** — it is the first
hard prerequisite for Phase 1.

### SPM consumability

`Package.swift` must sit at repo root. mlx-swift must pin **`exact: 0.31.6`** — the same pin
MPNNKit, BoltzMLX and RFD3MLX all carry. `exact:` has no consumer-side override, so a divergent
pin fails resolution outright and the app will not build.

---

## 4. Precision choice

**int8 group-64**, per the skill's default. ~307 M inference-resident params → **~310 MB** vs
1.23 GB bf16. Quantized matmul is also *faster* on these memory-bound ops.

Two RFD4-specific cautions:

- The checkpoint is stored **FP32 on disk (1.74 GB)** but the model *runs* BF16. So "full
  precision" is a narrowing choice, exactly as the skill warns.
- The decoder masks logits with `torch.finfo(dtype).min`. bf16 vs fp32 `finfo.min` differ —
  pin this in the parity harness or the 147-way residue readout drifts.

**Ship the AE encoder.** It is load-bearing only for partial diffusion / inverse folding —
for pure generation `x_1` is never read, since `full_simulation` samples from noise — so a
de-novo-only port could drop 129 M params (~30%). This port keeps it: upstream's
`examples/antibody_design/partial.yaml` is a partial-diffusion run, and the agreed scope
includes porting the examples. Note the encoder is **stochastic** (`encoder.py:147`
reparameterises), so parity must pin mean-vs-sample explicitly.

## 5. Scope & effort

Reuse is unusually good — your existing checkouts already carry most primitives:

| Need | Already have |
|---|---|
| Triangle multiplication | `protenix-mlx/Sources/ProtenixMLX/Layers/TriangleMultiplication.swift` |
| Pair stack / pairformer idiom | `boltz-mlx`, `protenix-mlx` `Trunk/Pairformer.swift` |
| Reference conformers (canonical-20) | `protenix-mlx` `Resources/residue_templates.json` |
| Sampler loop, budget guard, weight pack, PDB parse, trajectory + cancel hooks | `rfd3-mlx` (1,838 Swift LOC total) |

**Does NOT need porting** (verified): triangle *attention* (dead code — only its own definition),
FlashAttention, cuEquivariance fused kernels (all have torch fallbacks that *are* the reference),
`torch.compile`/CUDA-graph plumbing, SE(3)/IPA (none), openbabel/rdkit (metrics only),
SASA/DSSP/interaction annotation (training-time design-task sampling only).

### Estimate for the chosen full-replacement scope

| Phase | Work | Estimate |
|---|---|---|
| 0 | **Science gate §1b** | 3–5 days |
| 1 | Weight export + fixtures (flat state dict, no reshaping) | 1 week |
| 2a | Trunk: PairBiasAttention → TransitionADALN → PairReprUpdate → 14-layer trunk | 2–3 weeks |
| 2b | AE decoder (12-layer, no pair track) + 147-class vocab plumbing | 1 week |
| 2c | Sampler (`full_simulation`, TSR schedules) — pure tensor math, ~250 lines | 3–5 days |
| 2d | **Featurizer incl. target conditioning** — the long pole | **3–4 weeks** |
| 3 | Parity (developed alongside 2, not after) | — |
| 4–5 | int8-vs-fp32 accuracy, perf, memory fit | 1–2 weeks |
| 6 | RayMol integration, tag, weights asset, CI | 1 week |

**≈ 9–13 weeks.** The featurizer is the long pole exactly as you predicted when choosing this
scope. Everything that makes it tractable is scope restriction: canonical-20, single target
chain, hotspots only — *not* the full condition-spec DSL (ligands, NCAAs, glycans, nucleic acids,
RASA/SS/hbond conditioning). That full DSL is the 6-month version and is almost entirely
AtomWorks reimplementation rather than MLX work.

**RFD3's featurizer is 187 Swift lines** because all-protein systems bypass the reference-conformer
machinery. **Do not assume RFD4 collapses the same way** — its AE is unified across proteins,
nucleic acids and small molecules. First featurizer task is to determine, empirically, how much
of the ~40-tensor contract fills with constants for canonical-20.

---

## 6. Parity landmines (pin these in the harness from day one)

- `NEG_INF = -1e4`, **not** `-inf` (`pair_bias_attn.py:40`).
- Fully-masked-row fixup: `masked_fill(~mask.any(-1,keepdim=True), 0.0)` (`:169`) — skip it and
  padded rows go NaN.
- QK-LayerNorm is on the **full inner dim (768)**, not per-head.
- `AdaptiveOutputScale` bias init `-2.0` → `sigmoid(-2) ≈ 0.119`.
- Recycle projections are **zero-initialised and bias-free**; `s`/`z` start as `zeros_like`.
- **No coordinate recycling** — unlike AlphaFold, `x_t` is unchanged across recycles.
- `recycle_remaining` conditioning is a **constant `[1.0]`** under `recycle.grad: last`. Hard-code it.
- `vf_to_score` asserts `t < 1`; `score_to_vf` asserts `t > 0`; schedule starts at `eps`, and
  `simulation_step` special-cases `t == 0.0`.
- `_apply_stochastic_offset` runs **once before step 0** — hoist to host-side numpy.
- Featurization calls `np.random.permutation` on chain/molecule ids. Semantically irrelevant
  (equality-only pair features) but **it breaks seed-level reproducibility against the reference** —
  neutralise it in the export harness or parity will chase a ghost.

---

## 7. Gate output

| Item | Verdict |
|---|---|
| Metric discriminates | **Inherited from the existing funnel — not a new risk** |
| **Token-length generalization to 638** | **UNKNOWN — blocking, see §1b** |
| Memory model | ~1.5–2.5 GB at n=638 (estimate); needs conservative measured fit + `RFD4Budget` |
| Dep resolution | Mac CPU oracle confirmed by reading; **pip resolution unproven, needs approval** |
| SPM consumability | Root `Package.swift`, mlx-swift `exact: 0.31.6` in lockstep with the other three |
| Precision | int8 g64, full bundle incl. AE encoder (0.44 GB) |

**Verdict: CONDITIONAL GO.** Engineering risk is lower than RFD3's was — the networks are two
plain transformer towers with no equivariance and no custom kernels, and the device cost looks
*better* than RFD3's. The open risk is not engineering, it is whether the model conditions
correctly 2.5× beyond its training length. Run §1b first; it can kill or reshape the
full-replacement scope for a few days of CPU time.

---

## Open decisions for you

1. **Approve installing upstream** in a throwaway venv (blocks Phase 1 entirely).
2. Confirm the §1b experiment is the right gate, and that reverting to the narrow de-novo slice
   is the agreed fallback if it fails.
3. Repo name + visibility: `javierbq/rfd4-proteina-mlx`, public MIT wrapper over BSD-3 upstream,
   matching the boltz/protenix/rfd3 pattern.
