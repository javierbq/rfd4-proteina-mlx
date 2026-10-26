#!/usr/bin/env python3
"""Dump (inputs, output) pairs for individual trunk modules.

`dump_trunk_fixtures.py` captures outputs only, which is enough to see *where* a
divergence starts but not to drive a test: to check one Swift module in isolation
you must feed it exactly the tensors upstream fed its PyTorch counterpart. This
records both sides for a small set of modules.

Scope is deliberately layer 0 plus the first pair update. Every trunk layer is the
same code with different weights, so proving one layer against real activations
proves the port of that layer type; proving fourteen only inflates the fixture.
The pair representation is [B, I, I, 384] — 98 KB at 8 tokens, 6 MB at 64 — so it
is captured once and referenced, not stored per case.

Usage:
    python dump_layer_fixtures.py <snapshot-dir> <out.json> [n_tokens]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import torch

from rfproteina.proteina import Proteina

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_trunk_fixtures import build_batch  # noqa: E402  (shares the batch contract)

# module path -> names for its positional forward() arguments
CASES = {
    "transformer_layers.0.mhba": ["x", "pair_rep", "cond", "mask"],
    "transformer_layers.0.transition": ["x", "cond", "mask"],
    "transformer_layers.0": ["x", "pair_rep", "cond", "mask"],
    "pair_update_layers.0": None,  # names discovered from arity
}


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
    ).decode()


def entry(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "b64": b64(t)}


def resolve(root, path: str):
    mod = root
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def main() -> int:
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    model = Proteina.from_pretrained(snap, device="cpu")
    net = model.nn
    net.eval()
    batch = build_batch(n_tokens, 0)

    captured: dict[str, dict] = {}
    handles = []

    def hook(name: str, argnames):
        def fn(_module, inputs, output):
            if name in captured:          # first call only; recycling re-enters
                return
            rec: dict = {"inputs": {}}
            for i, t in enumerate(inputs):
                if not torch.is_tensor(t):
                    continue
                key = argnames[i] if argnames and i < len(argnames) else f"arg{i}"
                rec["inputs"][key] = entry(t)
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                rec["output"] = entry(tensor)
            captured[name] = rec
        return fn

    for path, argnames in CASES.items():
        try:
            mod = resolve(net, path)
        except Exception as exc:
            print(f"  skip {path}: {exc}")
            continue
        if mod is None:
            continue
        handles.append(mod.register_forward_hook(hook(path, argnames)))

    with torch.no_grad():
        net(batch)
    for h in handles:
        h.remove()

    # Deduplicate the heavy pair representation: identical tensors are stored once.
    shared: dict[str, dict] = {}
    for name, rec in captured.items():
        for key, val in list(rec["inputs"].items()):
            if len(val["b64"]) < 40000:
                continue
            digest = str(hash(val["b64"]))
            if digest not in shared:
                shared[digest] = val
            rec["inputs"][key] = {"ref": digest}

    payload = {
        "note": "base64 little-endian float32. Generated; do not hand-edit.",
        "torch": torch.__version__,
        "snapshot": snap.name,
        "n_tokens": n_tokens,
        "heads": net.transformer_layers[0].mhba.mha.heads,
        "dim_head": net.transformer_layers[0].mhba.mha.dim_head,
        "shared": shared,
        "cases": captured,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    for name, rec in captured.items():
        ins = ", ".join(
            k + ("(ref)" if "ref" in v else str(tuple(v["shape"])))
            for k, v in rec["inputs"].items()
        )
        print(f"  {name}: {ins} -> {tuple(rec.get('output', {}).get('shape', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
