#!/usr/bin/env python3
"""Capture upstream's initial state and every noise draw, for a matched-noise run.

Two independent samples of a 16-residue de-novo design differ a lot, so comparing
their statistics cannot distinguish a real divergence from sampling variance. This
records the reference's starting point and the exact Gaussian it draws at each
step, so the Swift sampler can be driven down the SAME trajectory and compared
pointwise.

torch.randn is wrapped only for the duration of each simulation_step call, so the
draws are attributed to their step and nothing else in the process is captured.

Usage: python dump_matched_noise.py <snapshot> <out.json> [n] [nsteps] [repo] [mode]
"""
import base64, json, sys
from functools import partial
from pathlib import Path
import torch
from omegaconf import OmegaConf
from rfproteina.proteina import Proteina
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_trunk_fixtures import build_batch

def b64(t): return base64.b64encode(t.detach().contiguous().to(torch.float32).numpy().tobytes()).decode()
def entry(t): return {"shape": list(t.shape), "b64": b64(t.to(torch.float32))}

def main():
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    repo = Path(sys.argv[5]) if len(sys.argv) > 5 else Path(
        "/private/tmp/claude-501/-Users-javier/"
        "b4675505-3fa2-5cb2-b5a1-0f318ce3a657/scratchpad/rfd4-src/RFD4-Proteina")
    mode = sys.argv[6] if len(sys.argv) > 6 else "flat"

    model = Proteina.from_pretrained(snap, device="cpu"); model.eval()
    gen_cfg = OmegaConf.load(repo / "configs/generation/inference_base.yaml")
    model.inf_cfg = gen_cfg
    batch = build_batch(n, 0)
    batch["padding_mask"] = torch.ones(1, n, dtype=torch.bool)

    if mode == "flat":
        depths = tuple(1 for _ in range(nsteps))
    else:
        bands = {"low": [[10, 4], [10, 3], [10, 2], [30, 1]]}[mode]
        depths = tuple(d for w, d in bands for _ in range(w))

    fm = model.fm
    real_step = fm.simulation_step
    real_randn = torch.randn
    steps, state = [], {"i": 0, "rec": None, "x0": None}

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
            out = real_step(x_t=x_t, **kw)
        finally:
            torch.randn = real_randn
        steps.append({"step": state["i"], "draws": state["rec"],
                      "x_next": {m: entry(v) for m, v in out.items()}})
        state["rec"] = None
        state["i"] += 1
        return out

    fm.simulation_step = traced_step
    try:
        with torch.no_grad():
            x, _ = fm.full_simulation(
                batch=batch, predict_for_sampling=partial(model.predict_for_sampling),
                recycle_depths=depths, nsteps=nsteps, nsamples=1, n=n,
                sampling_model_args=gen_cfg["model"], device=torch.device("cpu"))
    finally:
        fm.simulation_step = real_step
        torch.randn = real_randn

    json.dump({"note": "base64 le float32. Generated.", "n": n, "nsteps": nsteps,
               "mode": mode, "depths": list(depths), "x0": state["x0"],
               "steps": steps, "x_final": {m: entry(v) for m, v in x.items()}},
              open(out, "w"), indent=1)
    drew = sum(1 for s in steps if s["draws"])
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  {len(steps)} steps, {drew} drew noise, "
          f"{sum(len(s['draws']) for s in steps)} draws total")
    return 0

raise SystemExit(main())
