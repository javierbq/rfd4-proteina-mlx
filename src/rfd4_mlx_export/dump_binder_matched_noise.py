#!/usr/bin/env python3
"""A whole conditioned trajectory, driven by upstream's own randomness.

The binder path is verified at the featurizer: given the same inputs, the port's
three factory outputs match upstream's. That says nothing about the 60 sampling
steps that follow, and the de-novo work showed why that gap matters — three real
defects lived there, each invisible to single-call parity, and comparing two
independent samples could neither confirm nor refute any of them because sampling
variance swamps the difference.

The fix there was to remove the randomness from the comparison: capture upstream's
initial state and every Gaussian it draws, replay them through the Swift sampler,
and compare trajectories that differ only in arithmetic. That reached 5.4e-06 over
60 steps. This does the same thing for a CONDITIONED two-chain run, which the
de-novo fixture cannot reach: the target's coordinates, its sequence and the
hotspots all enter through features that are identically zero without them.

The problem is deliberately small — `"20, /0, A17-50"`, 54 tokens — because the
pair track is quadratic and a 225-token reference run on CPU is hours. The size is
not what is being tested; the conditioning is.

Usage: python dump_binder_matched_noise.py <snapshot> <spec.yaml> <out.json> [nsteps] [repo]
"""
from __future__ import annotations

import base64
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rfproteina.datapipes.collate import collate_batch
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.inference_utils.datasets import ConditionSpecDataset
from rfproteina.proteina import Proteina

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_binder_batch import FIXTURE_KEYS, describe_layout, entry  # noqa: E402

DEFAULT_REPO = Path(
    "/private/tmp/claude-501/-Users-javier/"
    "b4675505-3fa2-5cb2-b5a1-0f318ce3a657/scratchpad/rfd4-src/RFD4-Proteina"
)


def main() -> int:
    snap, spec, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    repo = Path(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_REPO

    ds = ConditionSpecDataset(
        data=spec,
        name="small-binder",
        prevalidate=False,
        transform=build_design_validation_pipeline(
            path_length_search_cutoff=64,
            expandable_segment_rng=np.random.default_rng(0),
        ),
    )
    batch = collate_batch(
        [ds[0]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample"
    )
    n = batch["padding_mask"].shape[-1]
    layout = describe_layout(batch)
    print(f"  {n} tokens, breaks at {layout.get('chain_breaks_at')}")

    model = Proteina.from_pretrained(snap, device="cpu")
    model.eval()
    gen_cfg = OmegaConf.load(repo / "configs/generation/inference_base.yaml")
    model.inf_cfg = gen_cfg

    # Depth 1 everywhere. The recycle schedule is already checked bitwise against
    # upstream's; holding it flat keeps this run tractable and keeps the comparison
    # about the conditioned featurization and the step maths.
    depths = tuple(1 for _ in range(nsteps))

    fm = model.fm
    real_step = fm.simulation_step
    real_randn = torch.randn
    steps: list = []
    state = {"i": 0, "rec": None, "x0": None}

    def spy_randn(*a, **k):
        t = real_randn(*a, **k)
        if state["rec"] is not None:
            state["rec"].append(entry(t))
        return t

    def traced_step(x_t, **kw):
        if state["x0"] is None:
            state["x0"] = {m: entry(v) for m, v in x_t.items()}
        state["rec"] = []
        torch.randn = spy_randn
        try:
            res = real_step(x_t=x_t, **kw)
        finally:
            torch.randn = real_randn
        steps.append(
            {
                "step": state["i"],
                "draws": state["rec"],
                # Every step's output, not just the last: a trajectory that diverges
                # slowly is a different defect from one that diverges at a particular
                # t, and only the per-step record tells them apart.
                "x_next": {m: entry(v) for m, v in res.items()},
            }
        )
        state["rec"] = None
        state["i"] += 1
        return res

    fm.simulation_step = traced_step
    try:
        with torch.no_grad():
            x, _ = fm.full_simulation(
                batch=batch,
                predict_for_sampling=partial(model.predict_for_sampling),
                recycle_depths=depths,
                nsteps=nsteps,
                nsamples=1,
                n=n,
                sampling_model_args=gen_cfg["model"],
                device=torch.device("cpu"),
            )
    finally:
        fm.simulation_step = real_step
        torch.randn = real_randn

    fields = {}
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value) and (
            key in FIXTURE_KEYS or key.startswith("condition_sequence-probability-")
        ):
            fields[key] = entry(value)

    from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

    prefix, suffix = "condition_sequence-probability-", "_1_residue"
    seq_prob_keys = [
        getattr(c, "full_name", str(c))[len(prefix): -len(suffix)]
        for c in SEQUENCE_PROBABILITY_CONDITIONS
        if getattr(c, "full_name", str(c)) in fields
    ]

    out.write_text(
        json.dumps(
            {
                "note": "base64 little-endian float32. Generated; do not hand-edit.",
                "source": str(spec),
                "n_tokens": int(n),
                "nsteps": nsteps,
                "depths": list(depths),
                "layout": layout,
                "seq_prob_keys": seq_prob_keys,
                "fields": fields,
                "x0": state["x0"],
                "steps": steps,
                "x_final": {m: entry(v) for m, v in x.items()},
            }
        )
    )
    drew = sum(1 for s in steps if s["draws"])
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(steps)} steps, {drew} drew noise, "
          f"{sum(len(s['draws']) for s in steps)} draws total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
