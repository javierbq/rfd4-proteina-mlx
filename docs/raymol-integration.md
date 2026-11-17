# Shipping RFD4 in RayMol

What this repo now provides, and what still has to be done on the RayMol side.

## The pack

Built with the `RFD4Export` target:

```
swift build -c release --product RFD4Export
./.build/release/RFD4Export \
  ~/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/*/model.safetensors \
  dist/rfd4-proteina-mlx-fp16 --precision fp16 --version v1
```

It narrows the checkpoint with `Weights.narrowed` — the same function the loader
calls, not a reimplementation of it — writes `rfd4_core.safetensors` plus
`manifest.json`, stamps provenance into the safetensors `__metadata__` as well as
the manifest, and then loads its own output back through `RFD4Pack` before
reporting success.

As built and PUBLISHED on 2026-11-17:

| | |
|---|---|
| source checkpoint | `model.safetensors`, sha256 `5b549dd6…c68fb0` |
| tensors / parameters | 1257 / 435,622,465 |
| precision | fp16 (see `validation/quality/fp16.md`) |
| `rfd4_core.safetensors` | 871,414,812 bytes, sha256 `e151e431…2cb3ef26` |
| zipped | 807,122,869 bytes, sha256 `c07d977c…8047a03d` |
| export time | 2.7 s |

**The pack is not byte-reproducible, by design.** `manifest.json` carries an
`exportedAt` timestamp, so every export yields a different archive digest. Any
sha256 recorded anywhere therefore belongs to ONE build — which is why the
bundle's digest is taken from the uploaded asset rather than from a rebuild.
The numbers above supersede an earlier 2026-11-11 build (871,414,786 /
807,137,648): commit a7a8fc5 changed the recorded `sourceCheckpoint` from the
symlink's target to its name, which moved the safetensors `__metadata__` by 26
bytes.

**Verified end to end**: a binder design generated from the pack directory is
byte-identical in its atom records to one generated from the bare checkpoint at
`RFD4_DTYPE=fp16`, same seed. The pack is exactly what loading the checkpoint
produces in memory, written down.

The zip must be FLAT. RayMol's `WeightCache._extract` compares the archive's
member set against the bundle's declared `members` and rejects any member that is
not a plain file name, so no subdirectories and no `__MACOSX`:

```
cd dist/rfd4-proteina-mlx-fp16 && zip -X ../rfd4-proteina-mlx-fp16-v1.zip \
    manifest.json rfd4_core.safetensors
```

## Loading

`Weights.loadPathOrPack(_:)` takes either a pack directory or a bare
`.safetensors` and decides by what is on disk, so a host that hands over a
directory and a CLI that hands over a file both work without either knowing which
the other does. `Generator(weightsAtPathOrPack:)` wraps it. The CLI uses it for
all three subcommands, which is why `RFD4MLXCLI binder <pack-dir> …` works.

A directory that is not a pack fails as a pack, naming the missing manifest.

## What the manifest pins, and why

`format` is probed before the rest is decoded, so a future pack fails with
"unsupported format" rather than a missing-key error.

`precision` is declared and passed to `Weights.load` explicitly, so `RFD4_DTYPE`
cannot reinterpret a pack that is already half.

`vocabulary` is the one that earns its place. `vocabulary.json` is compiled into
`RFD4Core` as a resource but is really part of the weights: the sequence head
emits one logit per residue class and the classes are identified by POSITION.
Regenerating it from a newer CCD snapshot can insert or reorder a class, after
which every residue past the change is named wrong — and that failure looks like
a success, producing a well-formed structure with the wrong chemistry. The pack
records name, snapshot and class count; a mismatch is an error.

`quantization` is reserved and absent. Declaring it now means an int8 pack later
is a `format` bump against a schema that already has somewhere to put the group
size, instead of a new schema that strands installed packs. A pack that sets it
is refused while there is no quantized matmul path, because running dense matmuls
over packed uint32 would produce noise rather than an error.

## Still to do, in the RayMol repo

This repo cannot do these; they are listed so the remaining work is visible.

1. ~~**Upload the zip** as a release asset, then hash **the asset you
   downloaded**, not the local build.~~ **DONE** (2026-11-17). The repo is
   public at `javierbq/rfd4-proteina-mlx`, tagged `0.1.0`, and the asset is
   released under `weights-rfd4-v1`. The digest below was taken from a fresh
   `curl` of the published URL and is byte-identical to the local build.
2. **`modules/pymol/generators/rfd4.py`** — an `RFD4Generator` modelled on
   `RFD3Generator`, carrying:

   ```python
   weight_bundle = WeightBundle(
       id='rfd4-proteina-mlx-fp16',
       version='v1',
       url='https://github.com/javierbq/rfd4-proteina-mlx/releases/download/'
           'weights-rfd4-v1/rfd4-proteina-mlx-fp16-v1.zip',
       sha256='c07d977c0c130f7760256f93787f73efde699e17cb526ec6ef935f5d8047a03d',
       size=807_122_869,
       members=('manifest.json', 'rfd4_core.safetensors'),
   )
   ```
3. **One line** in `generators/__init__.py:_register_builtins`.
4. **A Swift `RFD4JobManager`** conforming to `InferenceRuntime`, plus the MLX
   cache-limit policy type registered under its own owner name, plus a size guard.
5. **One line** in `InferenceRouter.runtimes`, macOS-filtered.
6. **`RAYMOL_PREDICT_RUNTIMES`** in `PyMOLBridge.mm` gains `rfd4` (macOS line
   only).
7. **`project.yml`** — a package entry pinning `mlx-swift` **exact 0.31.6** to
   match MPNNKit, BoltzMLX and RFD3MLX, plus the two target-dependency entries.
   The package resolves from `https://github.com/javierbq/rfd4-proteina-mlx.git`
   at `0.1.0`.

### The size guard needs these numbers

From `validation/benchmark/peak-memory.md`, measured on the real trunk types:

```
fp16   peak(n) = 1.56318e-05 n^2 - 8.14674e-04 n + 0.780   GB
```

638 tokens is 6.62 GB at fp16 (13.24 at fp32). Use the fp16 curve — that is what
ships.

## Why fp16 and not int8

int8 would be roughly another 1.9x smaller, but for RFD4 it needs an exporter
that does not exist, a quantized matmul path, a type change in every layer file,
and a decision about the five fused triangle-multiplication projections, which
are concatenated at construction and so have no single checkpoint tensor to
quantize. It also wants half-precision activations to be worth anything, so the
fp16 work is its prerequisite rather than its alternative. fp16 first is the
lower-risk order, and a second bundle can be added later beside this one without
touching the cache, the router or the UI.
