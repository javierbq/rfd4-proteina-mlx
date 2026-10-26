#!/usr/bin/env python3
"""Dump module-boundary activations from the flow trunk, for layer-by-layer parity.

The playbook develops parity *alongside* the port, not after: each Swift module is
checked against upstream at its own boundary, so a divergence is attributed to the
layer that caused it instead of surfacing as a wrong structure 14 layers later.

Deliberately small (default 8 tokens). These fixtures are committed, and the pair
representation is [B, I, I, 384] — at 8 tokens that is 98 KB per capture, at 64 it
would be 6 MB. Parity does not need large inputs; it needs exact ones.

What is captured:
  - the synthetic inputs, so the Swift side drives the identical batch
  - `s` in full at every layer boundary (the single representation, [B, I, 768])
  - `z` as statistics plus a leading slice -- full pair reps would dominate the file
    and a divergence in `z` shows up in `s` on the next layer anyway
  - all four head outputs in full

Values are base64 of raw little-endian float32, not decimal text, so nothing is lost
in a round trip.

Usage:
    python dump_trunk_fixtures.py <snapshot-dir> <out.json> [n_tokens]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import torch

from rfproteina.datapipes.schema.proteina import SPECS, TokenType
from rfproteina.proteina import Proteina

# Keep the pair-rep slice small; it is a tripwire, not the primary signal.
Z_SLICE = 64


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
    ).decode()


def tensor_entry(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "b64": b64(t)}


def stats_entry(t: torch.Tensor) -> dict:
    f = t.detach().to(torch.float32)
    return {
        "shape": list(f.shape),
        "mean": f.mean().item(),
        "std": f.std().item(),
        "min": f.min().item(),
        "max": f.max().item(),
        "head_b64": b64(f.flatten()[:Z_SLICE]),
    }


def build_batch(n_tokens: int, seed: int) -> dict:
    """A deterministic synthetic batch covering the schema's zero-filled contract.

    Every SPECS key is filled with its declared default, which is what `collate`
    does for absent keys -- so this is a batch the model accepts as well-formed without
    any of the AtomWorks featurization the port has not written yet.
    """
    g = torch.Generator().manual_seed(seed)
    batch: dict = {}
    for name, spec in SPECS.items():
        dtype, default, dims = spec[0], spec[1], spec[2]
        shape = [1]
        for d in str(dims).replace(",", " ").split():
            shape.append(n_tokens if d == "I" else int(d))
        if len(shape) == 1:
            shape.append(n_tokens)
        batch[name] = torch.full(shape, default, dtype=dtype)

    batch["padding_mask"] = torch.ones(1, n_tokens, dtype=torch.bool)
    batch["token_type"] = torch.full((1, n_tokens), TokenType.PROTEIN, dtype=torch.long)
    batch["x_t"] = {
        "bb_ca": torch.randn(1, n_tokens, 3, generator=g),
        "local_latents": torch.randn(1, n_tokens, 32, generator=g),
    }
    batch["t"] = {"bb_ca": torch.full((1,), 0.5), "local_latents": torch.full((1,), 0.5)}
    batch["n_recycle"] = 1
    return batch


def main() -> int:
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seed = 0

    model = Proteina.from_pretrained(snap, device="cpu")
    net = model.nn
    net.eval()

    batch = build_batch(n_tokens, seed)

    # Capture at each transformer layer and each live pair update.
    captures: dict[str, dict] = {}
    handles = []

    def hook(name: str):
        def fn(_module, _inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            # s is [B, I, D]; z is [B, I, I, D].
            captures[name] = (tensor_entry(tensor) if tensor.dim() <= 3
                              else stats_entry(tensor))
        return fn

    for i, layer in enumerate(net.transformer_layers):
        handles.append(layer.register_forward_hook(hook(f"transformer_layers.{i}")))
        handles.append(layer.mhba.register_forward_hook(hook(f"transformer_layers.{i}.mhba")))
        handles.append(
            layer.transition.register_forward_hook(hook(f"transformer_layers.{i}.transition"))
        )
    for i, pu in enumerate(net.pair_update_layers):
        if pu is not None:
            handles.append(pu.register_forward_hook(hook(f"pair_update_layers.{i}")))

    with torch.no_grad():
        out_dict = net(batch)

    for h in handles:
        h.remove()

    heads = {}
    for k, v in out_dict.items():
        if torch.is_tensor(v):
            heads[k] = tensor_entry(v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if torch.is_tensor(v2):
                    heads[f"{k}.{k2}"] = tensor_entry(v2)

    payload = {
        "note": "base64 little-endian float32. Generated; do not hand-edit.",
        "torch": torch.__version__,
        "snapshot": snap.name,
        "n_tokens": n_tokens,
        "seed": seed,
        "n_recycle": batch["n_recycle"],
        "inputs": {
            "x_t.bb_ca": tensor_entry(batch["x_t"]["bb_ca"]),
            "x_t.local_latents": tensor_entry(batch["x_t"]["local_latents"]),
            "t.bb_ca": tensor_entry(batch["t"]["bb_ca"]),
            "t.local_latents": tensor_entry(batch["t"]["local_latents"]),
        },
        "boundaries": captures,
        "heads": heads,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(f"  {len(captures)} module boundaries, {len(heads)} heads, n_tokens={n_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
