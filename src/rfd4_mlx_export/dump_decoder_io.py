#!/usr/bin/env python3
"""Capture the AE decoder's own inputs and outputs.

Same seam as `dump_trunk_io.py`, one network down: the decoder's first transformer
layer receives exactly `seq_init * attn_mask` and the conditioning vector, which is
everything its feature factories produce. Hooking there lets the Swift decoder be
driven without porting those factories first.

Usage:
    python dump_decoder_io.py <snapshot-dir> <out.json> [n_tokens]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import torch

from rfproteina.proteina import Proteina


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
    ).decode()


def entry(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "b64": b64(t.to(torch.float32))}


def main() -> int:
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    model = Proteina.from_pretrained(snap, device="cpu")
    dec = model.autoencoder.decoder
    dec.eval()

    g = torch.Generator().manual_seed(0)
    batch = {
        "z_latent": torch.randn(1, n, 32, generator=g),
        "ca_coors_nm": torch.randn(1, n, 3, generator=g),
        "padding_mask": torch.ones(1, n, dtype=torch.bool),
        "is_in_context_conditioning": torch.zeros(1, n, dtype=torch.bool),
        "supports_local_latent": torch.ones(1, n, dtype=torch.bool),
        "token_type": torch.zeros(1, n, dtype=torch.long),   # protein
    }

    captured: dict = {}

    def hook(_m, inputs, _o):
        if "seq_init" in captured:
            return
        seqs, _pair, cond, attn_mask = inputs[:4]
        captured["seq_init"] = entry(seqs)
        captured["cond"] = entry(cond)
        captured["attn_mask"] = entry(attn_mask.to(torch.float32))

    h = dec.transformer_layers[0].register_forward_hook(hook)
    with torch.no_grad():
        out_dict = dec(batch)
    h.remove()

    payload = {
        "note": "base64 little-endian float32. Generated; do not hand-edit.",
        "torch": torch.__version__,
        "snapshot": snap.name,
        "n_tokens": n,
        "inputs": {
            **captured,
            "ca_coors_nm": entry(batch["ca_coors_nm"]),
            "emit_mask": entry(out_dict["emit_mask"].to(torch.float32)),
            "token_type": entry(batch["token_type"].to(torch.float32)),
        },
        "outputs": {
            "coors_nm": entry(out_dict["coors_nm"]),
            "seq_logits": entry(out_dict["seq_logits"]),
            "aatype_max": entry(out_dict["aatype_max"].to(torch.float32)),
            "atom_mask": entry(out_dict["atom_mask"].to(torch.float32)),
            "atom_presence_logits": entry(out_dict["atom_presence_logits"]),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print("  inputs :", {k: tuple(v["shape"]) for k, v in payload["inputs"].items()})
    print("  outputs:", {k: tuple(v["shape"]) for k, v in payload["outputs"].items()})
    # A sanity figure for the write-up: how many atoms the decoder actually emitted.
    print(f"  atoms emitted: {int(out_dict['atom_mask'].sum())} over {n} residues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
