#!/usr/bin/env python3
"""Dump the batch upstream's own pipeline builds for a real binder design.

This is the specification for the Swift side's token layout. The port can already
compute every conditioning feature bit-exactly, but nothing yet turns a design
request -- a target structure plus "design 50-120 residues against it" -- into the
static fields those features read. Guessing them is exactly the mistake that made
de-novo generation produce a cloud of fragments, so they are measured here instead.

Runs the shipped PD-L1 spec (`docs/how-to/input_prep/examples/condition_spec_ppi.yaml`,
contig `50-120, /0, A17-131`) through `ConditionSpecDataset` + the design-validation
pipeline, and writes every static batch field plus a readable summary of the token
layout: which tokens are the designed chain, which are the target, where the chain
break falls, and how the connectivity fields express it.

It also runs the network once and captures the three factory outputs, so the Swift
side can be checked end to end on a real binder rather than only feature by
feature. That distinction has already cost this port once: every individual feature
matched while the concatenation did not, because one factory projection spans the
full width including the inert blocks.

Usage: python dump_binder_batch.py <snapshot-dir> <repo> <out.json> [seed] [--trim]
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
from rfproteina.inference_utils.datasets import ConditionSpecDataset

SPEC_REL = "docs/how-to/input_prep/examples/condition_spec_ppi.yaml"

# Anything above this many elements is stored as a summary instead of raw bytes;
# the point is the layout, not another copy of the coordinates.
MAX_ELEMS = 4_000_000


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.detach().contiguous().to(torch.float32).numpy().tobytes()
    ).decode()


def entry(t: torch.Tensor) -> dict:
    t = t if torch.is_tensor(t) else torch.as_tensor(t)
    f = t.to(torch.float32)
    rec: dict = {"shape": list(t.shape), "dtype": str(t.dtype)}
    if f.numel() <= MAX_ELEMS:
        rec["b64"] = b64(f)
    finite = f[torch.isfinite(f)]
    rec["finite"] = int(finite.numel())
    if finite.numel():
        rec["min"] = float(finite.min())
        rec["max"] = float(finite.max())
    return rec


def describe_layout(batch: dict) -> dict:
    """The token layout in plain terms, so a mismatch is readable rather than a diff."""
    n = batch["padding_mask"].shape[-1]
    out: dict = {"n_tokens": int(n)}

    def arr(key):
        v = batch.get(key)
        return None if v is None or not torch.is_tensor(v) else v[0].cpu().numpy()

    pl = arr("token_bond_path_length")
    cpo = arr("canonical_polymer_ordering")
    if pl is not None:
        # A chain break shows up as a zero path length between consecutive tokens:
        # 0 means disconnected, NOT adjacent.
        consec = np.array([pl[i + 1, i] for i in range(n - 1)])
        breaks = (np.flatnonzero(consec == 0) + 1).tolist()
        out["chain_breaks_at"] = breaks
        out["distinct_path_lengths"] = int(len(np.unique(pl)))
        out["path_length_max"] = int(pl.max())
    if cpo is not None:
        out["ordering_values"] = sorted(int(v) for v in np.unique(cpo))
        out["ordering_nonzero"] = int((cpo != 0).sum())

    for key in ("molecule_iid", "molecule_entity_id", "tok_idx", "token_type",
                "chain_iid", "chain_entity_id"):
        a = arr(key)
        if a is None:
            continue
        vals, counts = np.unique(a, return_counts=True)
        out[key] = {str(int(v)): int(c) for v, c in zip(vals, counts, strict=False)}

    # Which tokens carry a conditioned coordinate: this is the designed/target split.
    for key in ("coord_cond_mask", "atom37_mask", "hotspot"):
        a = arr(key)
        if a is None:
            continue
        per_token = (
            np.isfinite(a).any(axis=tuple(range(1, a.ndim)))
            if a.dtype.kind == "f"
            else (a != 0).any(axis=tuple(range(1, a.ndim)))
        ) if a.ndim > 1 else (np.isfinite(a) if a.dtype.kind == "f" else a != 0)
        idx = np.flatnonzero(per_token)
        out[f"{key}_tokens"] = {
            "count": int(idx.size),
            "first": int(idx[0]) if idx.size else None,
            "last": int(idx[-1]) if idx.size else None,
            "contiguous": bool(idx.size and idx[-1] - idx[0] + 1 == idx.size),
        }
    return out


#: The batch fields a Swift-built design problem has to reproduce. Everything else
#: in the 155-field batch is either padding-contract boilerplate or read by no live
#: feature, and a 7 MB fixture that is 95% unread is a fixture nobody checks.
FIXTURE_KEYS = [
    # layout + connectivity
    "padding_mask", "token_type", "tok_idx", "chain_iid", "molecule_iid",
    "molecule_entity_id", "condition_index-group_1_token",
    "token_bond_path_length", "canonical_polymer_ordering", "bond_order",
    # emit-mask inputs
    "is_in_context_conditioning", "supports_local_latent", "is_resolved_mask",
    # conditioning payload
    "condition_CoordinateGroup_1_atom", "condition_InternalCoordinate_1_atom",
    "condition_coordinate_1_atom", "condition_hotspot_1_atom", "atom37_mask",
    "condition_cath-label_1_residue", "residue_type",
]

FACTORIES = ("init_repr_factory", "cond_factory", "pair_repr_builder")

#: The pair representation is [1, 225, 225, 384] -- 78 MB raw, so a corner is stored
#: instead. A misaligned concatenation shows up in the first token pair as clearly as
#: in the last, and the de-novo fixture already checks the full array at 8 tokens.
PAIR_CORNER = 16


def capture_factories(net, batch: dict) -> tuple[dict, list]:
    """Run the net once, recording each factory's output and each creator's liveness."""
    outs: dict = {}
    rows: list = []
    handles = []

    def factory_hook(name):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if torch.is_tensor(t):
                outs[name] = t.detach().float()
        return fn

    def creator_hook(fname, cname):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if torch.is_tensor(t):
                a = t.detach().float()
                # The sequence track's creators are stored in full: that factory is
                # where every conditioning feature lands, so a disagreement there has
                # to be bisected to one creator and the alternative is guessing.
                # They are small ([1, n, w], 897 columns in all) and the pair track's
                # are not.
                rows.append((fname, cname, float(a.abs().max()),
                             entry(a) if fname == "init_repr_factory" else None))
        return fn

    for fname in FACTORIES:
        f = getattr(net, fname, None)
        if f is None:
            continue
        handles.append(f.register_forward_hook(factory_hook(fname)))
        creators = getattr(f, "feat_creators", None)
        if creators is None:
            inner = getattr(f, "init_repr_factory", None) or getattr(f, "cond_factory", None)
            creators = getattr(inner, "feat_creators", []) if inner else []
        for c in creators:
            handles.append(c.register_forward_hook(creator_hook(fname, type(c).__name__)))

    with torch.no_grad():
        net(batch)
    for h in handles:
        h.remove()
    return outs, rows


def main() -> int:
    snap, repo, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("-") else 0
    trim = "--trim" in sys.argv
    spec_path = repo / SPEC_REL

    pipeline = build_design_validation_pipeline(
        path_length_search_cutoff=64,
        # The contig asks for 50-120 residues; the design length is drawn from that
        # range, so it is pinned here or the fixture is a different problem each run.
        expandable_segment_rng=np.random.default_rng(seed),
    )
    ds = ConditionSpecDataset(
        data=spec_path, transform=pipeline, name="binder", prevalidate=False
    )
    print(f"  spec: {spec_path}")
    print(f"  examples: {len(ds)} ({', '.join(ds.names)})")
    sample = ds[0]
    batch = collate_batch(
        [sample], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample"
    )
    n = batch["padding_mask"].shape[-1]
    print(f"  tokens: {n}")

    def wanted(key: str) -> bool:
        if not trim:
            return True
        return key in FIXTURE_KEYS or key.startswith("condition_sequence-probability-")

    fields = {}
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value) and wanted(key):
            fields[key] = entry(value)
        elif isinstance(value, dict) and not trim:
            for sub, sv in sorted(value.items()):
                if torch.is_tensor(sv):
                    fields[f"{key}.{sub}"] = entry(sv)

    layout = describe_layout(batch)
    for k, v in layout.items():
        if k != "tok_idx":          # 225 singleton counts say nothing a range does not
            print(f"    {k}: {v}")

    # The trunk needs a noised state and a time; a plausible mid-trajectory one.
    g = torch.Generator().manual_seed(seed)
    batch["x_t"] = {
        "bb_ca": torch.randn(1, n, 3, generator=g),
        "local_latents": torch.randn(1, n, 32, generator=g),
    }
    batch["t"] = {"bb_ca": torch.full((1,), 0.5), "local_latents": torch.full((1,), 0.5)}
    batch["n_recycle"] = 1
    fields["x_t.bb_ca"] = entry(batch["x_t"]["bb_ca"])
    fields["x_t.local_latents"] = entry(batch["x_t"]["local_latents"])
    fields["t.bb_ca"] = entry(batch["t"]["bb_ca"])
    fields["t.local_latents"] = entry(batch["t"]["local_latents"])

    from rfproteina.proteina import Proteina

    model = Proteina.from_pretrained(snap, device="cpu")
    model.eval()
    outs, rows = capture_factories(model.nn, batch)

    factories = {}
    for name, t in outs.items():
        if t.dim() == 4:                      # the pair representation
            factories[name] = entry(t[:, :PAIR_CORNER, :PAIR_CORNER, :])
            factories[name]["corner"] = PAIR_CORNER
            factories[name]["full_shape"] = list(t.shape)
        else:
            factories[name] = entry(t)
        print(f"    factory {name}: {list(t.shape)}")

    creators = {}
    live = 0
    offset = 0
    for fname, cname, mx, payload in rows:
        rec = {"max": mx, "dead": mx == 0.0}
        if payload is not None:
            rec["output"] = payload
            rec["offset"] = offset
            offset += payload["shape"][-1]
        creators[f"{fname}/{cname}"] = rec
        live += mx != 0.0
    print(f"    creators: {live} live of {len(rows)}, seq width {offset}")

    # ORDER FROM THE REGISTRY, never sorted. The factory projects one matrix across
    # the concatenation, so a permuted channel list misaligns every weight after the
    # first difference while each individual value still looks right. Sorting these
    # as strings interleaves the amino acids with the nucleotides -- `A`, `ALA`,
    # `ARG`, `ASN`, `ASP`, `C`, ... -- and costs 8% of the sequence representation.
    from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

    prefix, suffix = "condition_sequence-probability-", "_1_residue"
    seq_prob_keys = []
    for cond in SEQUENCE_PROBABILITY_CONDITIONS:
        name = getattr(cond, "full_name", str(cond))
        if name in fields:
            seq_prob_keys.append(name[len(prefix): -len(suffix)])
    present = {k[len(prefix): -len(suffix)] for k in fields if k.startswith(prefix)}
    assert set(seq_prob_keys) == present, (
        f"registry order missed {sorted(present - set(seq_prob_keys))}"
    )

    out.write_text(
        json.dumps(
            {
                "source": str(spec_path.relative_to(repo)),
                "contig": "50-120, /0, A17-131",
                "seed": seed,
                "n_tokens": int(n),
                "layout": layout,
                "seq_prob_keys": seq_prob_keys,
                "fields": fields,
                "factories": factories,
                "creators": creators,
            }
        )
    )
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(fields)} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
