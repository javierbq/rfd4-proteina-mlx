#!/usr/bin/env python3
"""Capture the trunk's own inputs and outputs, so the whole 14-layer loop can be
checked without the featurizer.

`_run_trunk` sits exactly on the seam between featurization and the network: its
arguments (`s_init`, `z_init`, `c`, `seq_mask`) are everything the feature
factories produce, and its result is everything the heads consume. Recording that
boundary lets the Swift trunk — layers, pair updates, recycling and all four heads —
be driven and verified long before a single feature is computed in Swift.

Usage:
    python dump_trunk_io.py <snapshot-dir> <out.json> [n_tokens]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import torch

from rfproteina.proteina import Proteina

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_trunk_fixtures import build_batch  # noqa: E402


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
    ).decode()


def entry(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "b64": b64(t)}


def main() -> int:
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    model = Proteina.from_pretrained(snap, device="cpu")
    net = model.nn
    net.eval()
    batch = build_batch(n_tokens, 0)

    record: dict = {}
    original = net._run_trunk

    def traced(input_, s_init, z_init, c, seq_mask, *args, **kwargs):
        s, z = original(input_, s_init, z_init, c, seq_mask, *args, **kwargs)
        if "inputs" not in record:
            record["inputs"] = {
                "s_init": entry(s_init),
                "z_init": entry(z_init),
                "cond": entry(c),
                "mask": entry(seq_mask.to(torch.float32)),
            }
            record["trunk_out"] = {"s": entry(s), "z": entry(z)}
        return s, z

    net._run_trunk = traced
    try:
        with torch.no_grad():
            nn_out = net(batch)
    finally:
        net._run_trunk = original

    if "inputs" not in record:
        raise SystemExit("_run_trunk was never called — the trace hook did not fire")

    # Emit masks come from the batch, not the network; the Swift trunk takes them
    # as arguments because deriving them belongs to the featurizer.
    from rfproteina.flow_matching.product_space_flow_matcher import compute_flow_emit_masks

    emit = compute_flow_emit_masks(batch)
    record["emit"] = {k: entry(v.to(torch.float32)) for k, v in emit.items()}

    heads = {}
    for k, v in nn_out.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if torch.is_tensor(v2):
                    heads[f"{k}.{k2}"] = entry(v2)
    record["heads"] = heads

    payload = {
        "note": "base64 little-endian float32. Generated; do not hand-edit.",
        "torch": torch.__version__,
        "snapshot": snap.name,
        "n_tokens": n_tokens,
        "n_recycle": batch["n_recycle"],
        **record,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print("  inputs:", {k: tuple(v["shape"]) for k, v in record["inputs"].items()})
    print("  trunk :", {k: tuple(v["shape"]) for k, v in record["trunk_out"].items()})
    print("  emit  :", {k: tuple(v["shape"]) for k, v in record["emit"].items()})
    print("  heads :", {k: tuple(v["shape"]) for k, v in heads.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
