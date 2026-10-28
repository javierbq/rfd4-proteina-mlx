#!/usr/bin/env python3
"""The autoencoder ENCODER: per-creator feature outputs, layer IO, and the latent head.

The encoder is what partial diffusion needs. The flow model works on `local_latents`,
so starting a trajectory from a real structure rather than from noise means first
turning that structure into latents — which is the encoder's whole job, and the one
piece of the released bundle this port has never run.

Structurally it is the decoder's twin: 12 layers, `use_attn_pair_bias: False` and
`update_pair_repr: False`, so there is no pair track at all. What differs is the
feature set — it reads the GROUND-TRUTH structure `x1`, not the noised `x_t` — and
the head, a LayerNorm and one bias-free projection to `2 * latent_dim` that splits
into a mean and a log-scale.

That head is **stochastic**: `z = mean + randn_like(log_scale) * exp(log_scale)`. A
parity fixture therefore captures `mean` and `log_scale`, which are deterministic,
and leaves the draw to be matched separately — exactly the mean-vs-sample
distinction the port must not blur.

The input must be a structure with no unresolved atoms in an occupied atom37 slot.
That is not a convenience: `coords * mask` propagates NaN where the mask is 1 and the
coordinate is not modelled, attention spreads it to every token in the first layer,
and the encoder returns an entirely NaN latent. Upstream does this too -- measured,
on its own shipped binder target, which has six such atoms -- and its own partial
diffusion never trips over it because that flow consumes a GENERATED complex, where
every atom the mask claims really exists. So the fixture uses one.

Usage: python dump_encoder.py <snapshot> <input.cif|spec.yaml> <out.json>
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
from rfproteina.proteina import Proteina

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_binder_batch import FIXTURE_KEYS, entry  # noqa: E402

#: Batch fields the encoder's own features read, beyond the shared layout set.
ENCODER_KEYS = [
    "coords", "atom37_mask", "element", "ground_truth_element", "charge",
    "incident_bond_type_counts",
    "chain_iid", "residue_type", "is_resolved_mask",
]


def main() -> int:
    snap, spec, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

    pipeline = build_design_validation_pipeline(
        path_length_search_cutoff=64,
        expandable_segment_rng=np.random.default_rng(0))
    if spec.suffix in (".yaml", ".yml", ".json"):
        ds = ConditionSpecDataset(data=spec, name="encoder", prevalidate=False,
                                  transform=pipeline)
    else:
        from rfproteina.inference_utils.datasets import DesignProblemDataset
        ds = DesignProblemDataset([str(spec)], transform=pipeline)
    batch = collate_batch(
        [ds[0]], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
    n = batch["padding_mask"].shape[-1]
    print(f"  {n} tokens")

    model = Proteina.from_pretrained(snap, device="cpu")
    model.eval()
    encoder = model.autoencoder.encoder

    # The encoder reads x1, not x_t, so it needs no time and no noised state. It does
    # need `coords_nm`, which is not a dataset field: the autoencoder's own wrapper
    # derives it (`autoencoder.py:188`) and the encoder's feature factory asserts on
    # it. The masks come from compute_ae_masks over fields already present.
    batch["coords_nm"] = batch["coords"] / 10.0

    # Refuse a degenerate input rather than capture a fixture of NaNs. An atom37 slot
    # the mask claims but the structure does not model makes the whole latent NaN.
    unresolved = int(
        ((~torch.isfinite(batch["coords"]).all(-1)) & batch["atom37_mask"].bool()).sum())
    if unresolved:
        raise SystemExit(
            f"  {unresolved} occupied atom37 slots have non-finite coordinates; the "
            "encoder would return an all-NaN latent. Use a generated complex.")
    rows, outs = [], {}
    handles = []

    def creator_hook(name):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if torch.is_tensor(t):
                rows.append((name, entry(t.detach().float())))
        return fn

    def grab(name):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if torch.is_tensor(t):
                outs[name] = entry(t.detach().float())
        return fn

    for c in encoder.init_repr_factory.feat_creators:
        handles.append(c.register_forward_hook(creator_hook(type(c).__name__)))
    handles.append(encoder.init_repr_factory.register_forward_hook(grab("init_repr_factory")))
    # The layer outputs are deliberately NOT captured. They are 691 KB each at this
    # size and nothing checks them: the block is the decoder's, already verified
    # layer by layer there, so a disagreement here would be in the features or the
    # head and both are captured. A fixture should hold what is compared.

    # Pin the reparameterisation draw so the sample is reproducible; `mean` and
    # `log_scale` are what parity is actually checked against.
    torch.manual_seed(0)
    with torch.no_grad():
        output = encoder(batch)
    for h in handles:
        h.remove()

    # Keyed by POSITION, not class name. `Atom37NanometersCoorsSeqFeat` appears twice
    # -- absolute and CA-relative, the same class with a different flag -- so keying
    # by name silently keeps only the second and the fixture loses a feature.
    offset = 0
    creators = {}
    for index, (name, rec) in enumerate(rows):
        rec["offset"] = offset
        rec["creator"] = name
        offset += rec["shape"][-1]
        creators[f"{index}:{name}"] = rec
        print(f"  {rec['offset']:5d} +{rec['shape'][-1]:5d}  {name}")
    print(f"  concatenated width {offset} "
          f"(linear_out expects {encoder.init_repr_factory.linear_out.weight.shape[1]})")

    fields = {}
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value) and (key in FIXTURE_KEYS or key in ENCODER_KEYS):
            fields[key] = entry(value)

    out.write_text(json.dumps({
        "note": "base64 little-endian float32. Generated; do not hand-edit.",
        "source": str(spec),
        "n_tokens": int(n),
        "seq_width": offset,
        "fields": fields,
        "creators": creators,
        "outputs": outs,
        "mean": entry(output["mean"].detach().float()),
        "log_scale": entry(output["log_scale"].detach().float()),
        "z_latent": entry(output["z_latent"].detach().float()),
        "emit_mask": entry(output["emit_mask"].detach().float()),
    }))
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
