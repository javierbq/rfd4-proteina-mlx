#!/usr/bin/env python3
"""Dump a REAL binder-design batch and every live feature's output.

The de-novo fixtures cannot verify conditioning: the four creators that carry a
target — its coordinates, its sequence, its hotspots, and backbone ordering — are
all identically zero without one. This runs upstream's own pipeline on a shipped
binder example and captures what they actually produce.

Large tensors are summarised rather than stored. The pair representation at 225
tokens is [1, 225, 225, 384] = 78 MB; per-feature pair outputs are narrow enough
to keep whole, and verifying the four new creators needs the features, not the
projection over them.

Usage:
    python dump_conditioned_fixtures.py <snapshot> <input.cif> <out.json>
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

SCHEMA = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"
MAX_BYTES = 4_000_000   # keep any single stored tensor under ~4 MB


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.detach().contiguous().to(torch.float32).numpy().tobytes()
    ).decode()


def entry(t: torch.Tensor) -> dict:
    a = t.detach().to(torch.float32)
    rec = {"shape": list(a.shape), "dead": bool(a.abs().max().item() == 0.0)}
    if a.numel() * 4 <= MAX_BYTES:
        rec["b64"] = b64(a)
    else:
        rec["summary"] = {
            "mean": a.mean().item(), "std": a.std().item(),
            "min": a.min().item(), "max": a.max().item(),
        }
        rec["head_b64"] = b64(a.flatten()[:256])
    return rec


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
    snap, cif, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    model = Proteina.from_pretrained(snap, device="cpu")
    model.eval()
    net = model.nn

    pipeline = build_design_validation_pipeline(
        path_length_search_cutoff=64,
        expandable_segment_rng=np.random.default_rng(0),
    )
    dataset = DesignProblemDataset([str(cif)], transform=pipeline)
    batch = collate_batch([dataset[0]], schema=SCHEMA)
    n = batch["padding_mask"].shape[-1]

    g = torch.Generator().manual_seed(0)
    batch["x_t"] = {
        "bb_ca": torch.randn(1, n, 3, generator=g),
        "local_latents": torch.randn(1, n, 32, generator=g),
    }
    batch["t"] = {"bb_ca": torch.full((1,), 0.5),
                  "local_latents": torch.full((1,), 0.5)}
    batch["n_recycle"] = 1

    feats: dict[str, dict] = {}
    handles = []

    def hook_for(key: str):
        def fn(_m, _i, output):
            # MUST return None: a forward hook that returns a value replaces the
            # module's output.
            t = output[0] if isinstance(output, tuple) else output
            if key not in feats and torch.is_tensor(t):
                feats[key] = entry(t)
        return fn

    for fname in ("init_repr_factory", "cond_factory", "pair_repr_builder"):
        seen: dict[str, int] = {}
        for c in creators_of(getattr(net, fname)):
            nm = type(c).__name__
            seen[nm] = seen.get(nm, 0) + 1
            suffix = "" if seen[nm] == 1 else f"#{seen[nm]}"
            handles.append(c.register_forward_hook(hook_for(f"{fname}/{nm}{suffix}")))

    with torch.no_grad():
        net(batch)
    for h in handles:
        h.remove()

    # Batch fields the live creators read. Names come from the condition registry
    # rather than string literals, so a rename fails here instead of silently
    # producing a zero feature.
    # The condition registry is split: the generic ones (coordinate and sequence
    # conditioning) belong to atomworks, the RFD4-specific ones to rfproteina.
    from atomworks.ml.conditions import C_CRD, C_SEQ
    from rfproteina.conditions import C_CATH, C_CRDGRP, C_HOT, C_IDXGRP

    wanted = {
        "token_type": "token_type",
        "padding_mask": "padding_mask",
        "cath_code": C_CATH.full_name,
        "coord_group": C_CRDGRP.full_name,
        "index_group": C_IDXGRP.full_name,
        "molecule_iid": "molecule_iid",
        "molecule_entity_id": "molecule_entity_id",
        "bond_order": "bond_order",
        "token_bond_path_length": "token_bond_path_length",
        "hotspot": C_HOT.full_name,
        "coord_cond_mask": C_CRD.full_name,
        "seq_cond": C_SEQ.full_name,
        "canonical_polymer_ordering": "canonical_polymer_ordering",
        "coords_nm": "coords_nm",
        "atom37_mask": "atom37_mask",
    }

    inputs = {
        "x_t.bb_ca": entry(batch["x_t"]["bb_ca"]),
        "x_t.local_latents": entry(batch["x_t"]["local_latents"]),
        "t.bb_ca": entry(batch["t"]["bb_ca"]),
        "t.local_latents": entry(batch["t"]["local_latents"]),
    }
    missing = []
    for key, src in wanted.items():
        if src in batch and torch.is_tensor(batch[src]):
            inputs[key] = entry(batch[src].to(torch.float32))
        else:
            missing.append(f"{key} <- {src}")

    # The collated batch is a PaddedBatchedDataSample, not a dict: `in` works but
    # iteration yields indices, so the key list comes from the registry.
    from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

    seq_prob = []
    for cond in SEQUENCE_PROBABILITY_CONDITIONS:
        name = getattr(cond, "full_name", str(cond))
        if name in batch and torch.is_tensor(batch[name]):
            seq_prob.append(name)
            inputs["seqprob/" + name] = entry(batch[name].to(torch.float32))

    payload = {
        "note": "base64 little-endian float32. Generated; do not hand-edit.",
        "torch": torch.__version__,
        "source": cif.name,
        "n_tokens": n,
        "inputs": inputs,
        "features": feats,
        "missing_batch_keys": missing,
        "seq_prob_keys": seq_prob,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    live = [k for k, v in feats.items() if not v["dead"]]
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB), {n} tokens")
    print(f"  {len(live)}/{len(feats)} creators live")
    for k in sorted(live):
        print(f"    live {k}")
    if missing:
        print(f"  batch keys NOT found: {'; '.join(missing)}")
    print(f"  sequence-probability keys: {len(seq_prob)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
