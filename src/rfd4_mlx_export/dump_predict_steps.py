#!/usr/bin/env python3
"""Dump (x_t, t, v) at several points of a real upstream run.

Isolates `predict` from `step`. Every single-call parity test passes, yet the
Swift loop collapses while upstream's does not on the same batch, so the fault is
in a sequence. Capturing the reference's own state and its predicted vector field
at several times lets the Swift side be driven from identical inputs and compared
one stage at a time.

Usage: python dump_predict_steps.py <snapshot-dir> <out.json> [n] [nsteps]
"""
import base64, json, sys
from functools import partial
from pathlib import Path
import torch
from omegaconf import OmegaConf
from rfproteina.proteina import Proteina
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_trunk_fixtures import build_batch

REPO = Path("/private/tmp/claude-501/-Users-javier/"
            "b4675505-3fa2-5cb2-b5a1-0f318ce3a657/scratchpad/rfd4-src/RFD4-Proteina")

def b64(t): return base64.b64encode(t.detach().contiguous().to(torch.float32).numpy().tobytes()).decode()
def entry(t): return {"shape": list(t.shape), "b64": b64(t.to(torch.float32))}

def main():
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    model = Proteina.from_pretrained(snap, device="cpu"); model.eval()
    gen_cfg = OmegaConf.load(REPO / "configs/generation/inference_base.yaml")
    model.inf_cfg = gen_cfg
    batch = build_batch(n, 0)
    batch["padding_mask"] = torch.ones(1, n, dtype=torch.bool)

    want = {0, 1, 30, 59}
    caught, counter = {}, {"i": 0}
    real = model.predict_for_sampling

    # Also capture the trunk's own inputs at those steps, to separate a wrong
    # featurization from a wrong trunk call.
    trunk_io = {}
    orig_run = model.nn._run_trunk
    def traced_trunk(input_, s_init, z_init, c, seq_mask, *a, **k):
        i = counter["i"]
        if i in want and str(i) not in trunk_io:
            trunk_io[str(i)] = {"s_init": entry(s_init), "z_init": entry(z_init),
                                "cond": entry(c), "mask": entry(seq_mask.to(torch.float32))}
        return orig_run(input_, s_init, z_init, c, seq_mask, *a, **k)
    model.nn._run_trunk = traced_trunk

    def traced(b, *a, **k):
        i = counter["i"]; counter["i"] += 1
        res = real(b, *a, **k)
        if i in want:
            caught[str(i)] = {
                "x_t.bb_ca": entry(b["x_t"]["bb_ca"]),
                "x_t.local_latents": entry(b["x_t"]["local_latents"]),
                "t.bb_ca": entry(b["t"]["bb_ca"]),
                "t.local_latents": entry(b["t"]["local_latents"]),
                "v.bb_ca": entry(res["bb_ca"]["v"]),
                "v.local_latents": entry(res["local_latents"]["v"]),
            }
        return res

    model.predict_for_sampling = traced
    try:
        with torch.no_grad():
            model.fm.full_simulation(
                batch=batch, predict_for_sampling=partial(model.predict_for_sampling),
                recycle_depths=tuple(1 for _ in range(nsteps)), nsteps=nsteps,
                nsamples=1, n=n, sampling_model_args=gen_cfg["model"],
                device=torch.device("cpu"))
    finally:
        model.predict_for_sampling = real

    emit = {}
    from rfproteina.flow_matching.product_space_flow_matcher import compute_flow_emit_masks
    for k, v in compute_flow_emit_masks(batch).items():
        emit[k] = entry(v.to(torch.float32))

    json.dump({"note": "base64 le float32. Generated.", "n": n, "nsteps": nsteps,
               "emit": emit, "steps": caught, "trunk_io": trunk_io},
              open(out, "w"), indent=1)
    print(f"wrote {out} with steps {sorted(caught)}")
    for k, v in sorted(caught.items(), key=lambda kv: int(kv[0])):
        import struct
        t = struct.unpack("<f", base64.b64decode(v["t.bb_ca"]["b64"])[:4])[0]
        print(f"  step {k:>2}: t={t:.5f}")
    return 0

raise SystemExit(main())
