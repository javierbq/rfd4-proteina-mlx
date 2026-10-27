#!/usr/bin/env python3
"""Run upstream's own full_simulation on the batch the Swift port synthesises.

The bisection for the collapsing-chain defect. Every piece of the Swift pipeline
matches upstream on a single call, so the fault is either the batch the port
builds for a de-novo request or the way it drives the pieces across steps. This
runs the reference implementation on the SAME batch and traces the same numbers:

  - if it collapses too, the batch is wrong
  - if it produces a protein, the wiring is wrong

Usage:
    python trace_upstream_run.py <snapshot-dir> [n_tokens] [nsteps]
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rfproteina.proteina import Proteina

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_trunk_fixtures import build_batch  # noqa: E402


def trace(label: str, ca: torch.Tensor) -> None:
    """Rg about the per-axis mean, and consecutive CA distances, in nanometres.

    Rg about a SCALAR mean conflates centroid offset with spread and makes a
    collapsing chain look healthy; that mistake is why this defect was missed once.
    """
    c = ca[0]                                   # [n, 3]
    centred = c - c.mean(dim=0, keepdim=True)
    rg = centred.pow(2).sum(-1).mean().sqrt()
    d = (c[1:] - c[:-1]).pow(2).sum(-1).sqrt()
    print(f"  {label:>9}  Rg {rg:.3f} nm   CA-CA mean {d.mean():.3f} nm "
          f"(min {d.min():.3f} max {d.max():.3f})")


def main() -> int:
    snap = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    model = Proteina.from_pretrained(snap, device="cpu")
    model.eval()
    # from_pretrained does not populate inf_cfg; the released generation config is
    # what the schedules and per-modality sampling parameters come from.
    repo = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(
        "/private/tmp/claude-501/-Users-javier/"
        "b4675505-3fa2-5cb2-b5a1-0f318ce3a657/scratchpad/rfd4-src/RFD4-Proteina")
    gen_cfg = OmegaConf.load(repo / "configs/generation/inference_base.yaml")
    model.inf_cfg = gen_cfg

    # The same batch the Swift port builds: a plain de-novo request, every static
    # field at the default the collate produces.
    batch = build_batch(n, 0)
    batch["padding_mask"] = torch.ones(1, n, dtype=torch.bool)

    fm = model.fm
    sampling_args = gen_cfg["model"]
    # Depth schedule: "flat" for the wiring probe, or a released strategy so the
    # comparison is against the same vector field the Swift run uses.
    mode = sys.argv[5] if len(sys.argv) > 5 else "flat"
    if mode == "flat":
        depths = tuple(1 for _ in range(nsteps))
    else:
        bands = {"low": [[10, 4], [10, 3], [10, 2], [30, 1]],
                 "medium": [[5, 6], [10, 5], [5, 4], [10, 3], [20, 2], [10, 1]]}[mode]
        depths = tuple(d for w, d in bands for _ in range(w))
        assert len(depths) == nsteps, f"{mode} expands to {len(depths)}, not {nsteps}"
    print(f"upstream full_simulation: n={n} nsteps={nsteps} "
          f"trunk passes={sum(max(1, d) for d in depths)}")

    x0 = {
        "bb_ca": torch.randn(1, n, 3, generator=torch.Generator().manual_seed(1)),
        "local_latents": torch.randn(1, n, 32, generator=torch.Generator().manual_seed(2)),
    }
    trace("start", x0["bb_ca"])

    seen: list[tuple[int, torch.Tensor]] = []
    real_step = fm.simulation_step

    def traced_step(*args, **kwargs):
        out = real_step(*args, **kwargs)
        return out

    fm.simulation_step = traced_step
    try:
        with torch.no_grad():
            x, info = fm.full_simulation(
                batch=batch,
                predict_for_sampling=partial(model.predict_for_sampling),
                recycle_depths=depths,
                nsteps=nsteps,
                nsamples=1,
                n=n,
                sampling_model_args=sampling_args,
                device=torch.device("cpu"),
                save_trajectory_every=10,
            )
    finally:
        fm.simulation_step = real_step

    trace("final", x["bb_ca"])
    steps = info.get("trajectory") if isinstance(info, dict) else None
    if steps:
        for snapshot in steps:
            if "x_1" in snapshot and "bb_ca" in snapshot["x_1"]:
                trace(f"step {snapshot['step']}", snapshot["x_1"]["bb_ca"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
