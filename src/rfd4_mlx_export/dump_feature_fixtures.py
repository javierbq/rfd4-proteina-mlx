#!/usr/bin/env python3
"""Dump every feature creator's output, so each can be ported and checked alone.

The three factories hold 35 feature creators between them, but only 13 carry
signal for an unconditioned canonical-20 protein; the other 22 are identically
zero in that regime and come alive only under conditioning. Porting all 35 blind
would be mostly dead code with no way to tell which half was wrong.

This records each creator's output for a known batch, plus the batch inputs they
read, so a Swift implementation can be checked one feature at a time. Creators
whose output is all zeros are recorded too, with a `dead` flag — asserting that a
feature really is inert in this regime is worth a test of its own.

Usage:
    python dump_feature_fixtures.py <snapshot-dir> <out.json> [n_tokens]
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

FACTORIES = ("init_repr_factory", "cond_factory", "pair_repr_builder")


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
    ).decode()


def entry(t: torch.Tensor) -> dict:
    a = t.detach().to(torch.float32)
    return {
        "shape": list(a.shape),
        "b64": b64(a),
        "dead": bool(a.abs().max().item() == 0.0),
    }


def creators_of(factory):
    cs = getattr(factory, "feat_creators", None)
    if cs is not None:
        return list(cs)
    for attr in ("init_repr_factory", "cond_factory"):
        inner = getattr(factory, attr, None)
        if inner is not None and hasattr(inner, "feat_creators"):
            return list(inner.feat_creators)
    return []


def main() -> int:
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    model = Proteina.from_pretrained(snap, device="cpu")
    net = model.nn
    net.eval()
    batch = build_batch(n_tokens, 0)

    records: dict[str, dict] = {}
    handles = []

    def hook(key: str):
        def fn(_m, _i, output):
            if key in records:
                return
            t = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(t):
                records[key] = entry(t)
        return fn

    for fname in FACTORIES:
        factory = getattr(net, fname, None)
        if factory is None:
            continue
        seen: dict[str, int] = {}
        for creator in creators_of(factory):
            cname = type(creator).__name__
            # TimeEmbeddingBatchFeat appears twice, once per modality.
            seen[cname] = seen.get(cname, 0) + 1
            suffix = "" if seen[cname] == 1 else f"#{seen[cname]}"
            handles.append(creator.register_forward_hook(hook(f"{fname}/{cname}{suffix}")))

    with torch.no_grad():
        net(batch)
    for h in handles:
        h.remove()

    # The batch values the live features read, so Swift can be driven identically.
    inputs = {
        "x_t.bb_ca": entry(batch["x_t"]["bb_ca"]),
        "x_t.local_latents": entry(batch["x_t"]["local_latents"]),
        "t.bb_ca": entry(batch["t"]["bb_ca"]),
        "t.local_latents": entry(batch["t"]["local_latents"]),
        "token_type": entry(batch["token_type"].to(torch.float32)),
        "padding_mask": entry(batch["padding_mask"].to(torch.float32)),
    }
    # The remaining batch fields the live features read. Names come from the
    # condition registry, so they are looked up rather than hard-coded.
    from rfproteina.conditions import C_CATH, C_CRDGRP, C_IDXGRP
    for key, src in (("cath_code", C_CATH.full_name),
                     ("coord_group", C_CRDGRP.full_name),
                     ("index_group", C_IDXGRP.full_name),
                     ("molecule_iid", "molecule_iid"),
                     ("molecule_entity_id", "molecule_entity_id"),
                     ("bond_order", "bond_order"),
                     ("token_bond_path_length", "token_bond_path_length")):
        if src in batch and torch.is_tensor(batch[src]):
            inputs[key] = entry(batch[src].to(torch.float32))
        else:
            print(f"  (batch has no {src!r} for {key})")

    # The three factory outputs, so the whole featurizer can be checked at once.
    outs = {}
    for fname in FACTORIES:
        f = getattr(net, fname, None)
        if f is None: continue
        rec = {}
        def grab(_m, _i, o, rec=rec):
            # MUST return None: a forward hook that returns a value REPLACES the
            # module's output, which silently feeds the fixture dict downstream.
            if "out" not in rec and torch.is_tensor(o):
                rec["out"] = entry(o)
        h = f.register_forward_hook(grab)
        with torch.no_grad():
            net(batch)
        h.remove()
        if "out" in rec: outs[fname] = rec["out"]

    payload = {
        "note": "base64 little-endian float32. Generated; do not hand-edit.",
        "torch": torch.__version__,
        "n_tokens": n_tokens,
        "inputs": inputs,
        "features": records,
        "factories": outs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    live = sum(0 if r["dead"] else 1 for r in records.values())
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(f"  {len(records)} creators, {live} live, {len(records)-live} inert in this regime")
    for key, r in records.items():
        if not r["dead"]:
            print(f"    live {key:60} {tuple(r['shape'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
