#!/usr/bin/env python3
"""A whole partial-diffusion run, driven by upstream's own randomness.

Partial diffusion starts from a structure instead of from noise: corrupt it to a
chosen time, then denoise the remaining tail of the trajectory. Four things have to
agree for that to reproduce, and only the last of them is shared with ordinary
generation.

  1. `x_1`, the clean state -- CA coordinates in nanometres, and the ENCODER's
     latents for the sequence. The encoder is why this flow needed porting first.
  2. `partial_t`. For the fractional form it is not a free parameter but a point
     on the configured schedule: `k = nsteps - round(frac * nsteps)` and
     `partial_t = ts[k]`. "A restart index is not a diffusion time" -- change the
     schedule and the same fraction means a different corruption.
  3. The corruption. `x_t = (1 - t) * x_0 + t * x_1`, both endpoints emit-masked --
     but `x_0` for `bb_ca` is not a plain Gaussian. Measured off the loaded model:
     `zero_com_noise=True`, then `stochastic_centering_mode="x_0"` with
     `scope="per_group"` and `scale=0.2`, which adds an independent random
     translation per group. `local_latents` has none of that.
  4. The tail schedule: the last `nsteps` intervals of the full one.

So this captures `x_1`, every Gaussian drawn during corruption and during the run,
and the state after each step. A port replaying those draws differs from upstream
only in arithmetic -- the same technique that settled the de-novo and binder
trajectories, where comparing independent samples could not.

Usage: python dump_partial_diffusion.py <snapshot> <input.cif> <out.json> [frac] [nsteps]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rfproteina.datapipes.collate import collate_batch
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.inference_utils.datasets import DesignProblemDataset
from rfproteina.proteina import Proteina
from rfproteina.search.vanilla import build_partial_schedule

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_binder_batch import FIXTURE_KEYS, entry  # noqa: E402
from dump_encoder import ENCODER_KEYS  # noqa: E402


def main() -> int:
    snap, cif, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4166666666666667
    nsteps = int(sys.argv[5]) if len(sys.argv) > 5 else 60

    ds = DesignProblemDataset([str(cif)], transform=build_design_validation_pipeline(
        path_length_search_cutoff=64,
        expandable_segment_rng=np.random.default_rng(0)))
    batch = collate_batch(
        [ds[0]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
    n = batch["padding_mask"].shape[-1]

    model = Proteina.from_pretrained(snap, device="cpu")
    model.eval()
    # from_pretrained leaves inf_cfg unset; the released generation config is where
    # the schedules and the per-modality sampling parameters come from.
    from omegaconf import OmegaConf
    repo = Path("/private/tmp/claude-501/-Users-javier/"
                "b4675505-3fa2-5cb2-b5a1-0f318ce3a657/scratchpad/rfd4-src/RFD4-Proteina")
    model.inf_cfg = OmegaConf.load(repo / "configs/generation/inference_base.yaml")
    fm = model.fm

    # The flow matcher's corruption settings, measured rather than read off a
    # training config the released checkpoint may not have used.
    settings = {}
    for dm, b in fm.base_flow_matchers.items():
        settings[dm] = {
            "dim": int(b.dim),
            "zero_com_noise": bool(b.zero_com_noise),
            "stochastic_centering_mode": str(b.stochastic_centering_mode),
            "stochastic_centering_scale": (
                None if b.stochastic_centering_scale is None
                else float(b.stochastic_centering_scale)),
            "stochastic_centering_scope": str(b.stochastic_centering_scope),
            "group_keys": list(getattr(b, "stochastic_centering_group_keys", []) or []),
        }
        print(f"  {dm}: {settings[dm]}")

    batch = model.preprocess_batch_for_generation(batch)
    batch = model.add_clean_samples(batch)
    print(f"  {n} tokens; x_1 modes {sorted(batch['x_1'])}")

    sampling_model_args = model.inf_cfg["model"]
    ref_ts, _ = fm.sample_schedule(nsteps=nsteps, sampling_model_args=sampling_model_args)
    remaining = round(frac * nsteps)
    k = nsteps - remaining
    partial_t = {dm: float(ref_ts[dm][k].item()) for dm in fm.data_modes}
    print(f"  frac {frac} -> {remaining} of {nsteps} steps, restart index {k}, "
          f"partial_t {partial_t}")

    # Capture every Gaussian the corruption draws, then every one the run draws.
    real_randn = torch.randn
    rec: list | None = None

    def spy(*a, **kw):
        t = real_randn(*a, **kw)
        if rec is not None:
            rec.append(entry(t))
        return t

    corruption_draws: list = []
    rec = corruption_draws
    torch.randn = spy
    torch.manual_seed(0)
    try:
        batched_t = {dm: torch.full((1,), v) for dm, v in partial_t.items()}
        batch = fm.corrupt_batch(batch, t=batched_t)
    finally:
        torch.randn = real_randn
        rec = None
    print(f"  corruption drew {len(corruption_draws)} tensors")

    # Snapshot NOW. `partial_simulation` writes the running state back into
    # `batch["x_t"]` on every step, so reading it after the run gives the end of the
    # trajectory rather than the corruption -- which looks like a plausible x_t and
    # silently makes the interpolation unverifiable.
    snapshot = {
        "x_1": {m: entry(v) for m, v in batch["x_1"].items()},
        "x_0": {m: entry(v) for m, v in batch["x_0"].items()},
        "x_t": {m: entry(v) for m, v in batch["x_t"].items()},
    }
    for dm in fm.data_modes:
        x0, x1 = batch["x_0"][dm], batch["x_1"][dm]
        lerp = (1 - partial_t[dm]) * x0 + partial_t[dm] * x1
        gap = float((batch["x_t"][dm] - lerp).abs().max())
        print(f"    {dm}: |x_t - lerp(x_0, x_1)| = {gap:.4e}"
              f"{'  (the per-group offsets)' if gap > 1e-5 else ''}")

    partial_ts, partial_gt = build_partial_schedule(
        partial_t_dict=partial_t, nsteps=remaining, fm=fm,
        sampling_model_args=sampling_model_args, full_nsteps=nsteps)

    steps: list = []
    state = {"i": 0}
    real_step = fm.simulation_step

    def traced_step(x_t, **kw):
        nonlocal rec
        rec = []
        torch.randn = spy
        try:
            res = real_step(x_t=x_t, **kw)
        finally:
            torch.randn = real_randn
        steps.append({"step": state["i"], "draws": rec,
                      "x_next": {m: entry(v) for m, v in res.items()}})
        rec = None
        state["i"] += 1
        return res

    fm.simulation_step = traced_step
    try:
        with torch.no_grad():
            x, _, _ = fm.partial_simulation(
                batch=batch, x=batch["x_t"],
                predict_for_sampling=model.predict_for_sampling,
                recycle_depths=tuple(1 for _ in range(remaining)),
                nsteps=remaining, ts=partial_ts, gt=partial_gt,
                simulation_step_params={
                    dm: sampling_model_args[dm]["simulation_step_params"]
                    for dm in fm.data_modes},
                device=torch.device("cpu"))
    finally:
        fm.simulation_step = real_step

    fields = {}
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value) and (key in FIXTURE_KEYS or key in ENCODER_KEYS):
            fields[key] = entry(value)

    out.write_text(json.dumps({
        "note": "base64 little-endian float32. Generated; do not hand-edit.",
        "source": str(cif),
        "n_tokens": int(n),
        "full_nsteps": nsteps,
        "fraction": frac,
        "restart_index": k,
        "nsteps": remaining,
        "partial_t": partial_t,
        "settings": settings,
        "fields": fields,
        "ts": {dm: entry(partial_ts[dm]) for dm in fm.data_modes},
        "gt": {dm: entry(partial_gt[dm]) for dm in fm.data_modes},
        **snapshot,
        "corruption_draws": corruption_draws,
        "steps": steps,
        "x_final": {m: entry(v) for m, v in x.items()},
    }))
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {len(steps)} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
