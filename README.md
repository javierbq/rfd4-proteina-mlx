# rfd4-proteina-mlx

A Swift + [MLX](https://github.com/ml-explore/mlx-swift) port of
[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) — the Institute for Protein
Design and NVIDIA's flow-matching model for joint sequence and all-atom biomolecular design —
for running on Apple silicon inside [RayMol](https://github.com/javierbq/RayMol).

**Status: early. Nothing generates a structure yet.** See `docs/phase0-gate.md` for the
feasibility assessment and `docs/test-port-manifest.md` for the plan.

## How this port is being built

Upstream is ~60k lines of Python, of which the *networks* are about 450 lines. Roughly 80% of
the work here is featurization — reimplementing what upstream gets from
[AtomWorks](https://pypi.org/project/atomworks/) and [biotite](https://www.biotite-python.org/).

So the port is driven by **upstream's own test suite as the executable specification**. Tests
are ported first, red, and the implementation is written to turn them green. The alternative —
writing the featurizer and eyeballing tensors afterwards — is how featurization ports go wrong,
and upstream has already written down what correct means.

Triage of the suite (`src/rfd4_mlx_export/triage_tests.py`, regenerate any time):

| verdict | files | test fns |
|---|---|---|
| PORT | 44 | 233 |
| DROP | 23 | 115 |

Dropped tests are not skipped work — they are upstream concerns with no counterpart in an
on-device inference port: SLURM launchers, Hydra config resolution, gradient coverage, CUDA
regression baselines, the cuEquivariance-vs-fallback kernel comparison (the fallback *is* what
we port), Lightning training callbacks, and the evaluation suite that needs AlphaFold2 /
AlphaFold3 / OpenFold3 as external oracles. Each is listed with its reason in the manifest.

## Layout

```
Package.swift                      SPM manifest, repo root (SPM has no subdir support)
Sources/
  RFD4Core/                        vocabulary, structures, featurizer — NO MLX
  RFD4MLX/                         flow trunk, AE decoder, sampler — needs MLX
src/rfd4_mlx_export/               Python: resource generation, weight export, fixtures
Tests/                             ported upstream tests + parity tests
validation/{benchmark,quality,device}/
docs/                              gate doc, test port manifest, upstream license
```

### Why `RFD4Core` has no MLX dependency

The featurization tests assert on index arrays and masks, not GPU tensors. Keeping that module
MLX-free means the bulk of the suite builds and runs in **milliseconds**, on CI, with no Metal
device — which is what makes test-first viable at this scale. `RFD4MLX` carries everything that
genuinely needs a tensor library.

## The vocabulary is part of the weights

`Sources/RFD4Core/Resources/vocabulary.json` is a mechanical transcription of upstream's
`resources/aa_rna_dna_glycan_ncaa.yaml` (147 classes, CCD snapshot 2026-01-06), produced by
`src/rfd4_mlx_export/gen_vocabulary.py`. Upstream's warning applies here unchanged:

> Checkpoints depend on the residue order and each residue's atom-slot order. Reordering either
> changes what the learned weights mean.

Never hand-edit it. Regenerate from the YAML:

```sh
python3 -I src/rfd4_mlx_export/gen_vocabulary.py \
    <upstream>/src/rfproteina/resources/aa_rna_dna_glycan_ncaa.yaml \
    Sources/RFD4Core/Resources/vocabulary.json
```

## Dependency pinning

`mlx-swift` is pinned `exact: 0.31.6`, matching MPNNKit, BoltzMLX and RFD3MLX. `exact:` has no
consumer-side override, so if any one of the four bumps independently, resolution fails outright
and RayMol will not build at all. Bump them together and re-tag.

## Build

```sh
swift build
swift test
```

## License

BSD 3-Clause, matching upstream. Substantial parts of this port are translations of
RFD4-Proteina's source rather than independent reimplementations, so it carries the same
terms and retains the Institute for Protein Design's copyright alongside its own. The
unmodified upstream license is at `docs/UPSTREAM-LICENSE-BSD-3-Clause.txt`.

Model weights are distributed by NVIDIA and IPD under their own terms and are not
redistributed here.
