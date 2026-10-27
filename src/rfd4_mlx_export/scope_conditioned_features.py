"""Which feature creators come alive for a REAL binder-design input?

The port implements 13 of 35 creators and zero-fills the other 22, on the basis
that they are inert for an unconditioned canonical-20 protein. That measurement
says nothing about a conditioned run, which is the actual target scope. This runs
upstream's own pipeline on the shipped protein_binder example and reports which
creators carry signal there -- the exact list Stage C has to implement.
"""
import sys, torch
from pathlib import Path
from omegaconf import OmegaConf
from rfproteina.proteina import Proteina
from rfproteina.inference_utils.datasets import DesignProblemDataset
from rfproteina.datapipes.collate import collate_batch
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
import numpy as np

snap, repo, cif = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
model = Proteina.from_pretrained(snap, device="cpu"); model.eval()
net = model.nn

pipeline = build_design_validation_pipeline(
    path_length_search_cutoff=64,
    expandable_segment_rng=np.random.default_rng(0),
)
ds = DesignProblemDataset([str(cif)], transform=pipeline)
print(f"  dataset: {len(ds)} item(s)")
sample = ds[0]
batch = collate_batch([sample],
    schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
n = batch["padding_mask"].shape[-1]
print(f"  tokens: {n}")

# The trunk needs a noised state and a time; supply a plausible mid-trajectory one.
g = torch.Generator().manual_seed(0)
batch["x_t"] = {"bb_ca": torch.randn(1, n, 3, generator=g),
                "local_latents": torch.randn(1, n, 32, generator=g)}
batch["t"] = {"bb_ca": torch.full((1,), 0.5), "local_latents": torch.full((1,), 0.5)}
batch["n_recycle"] = 1

rows, handles = [], []
def mk(fname, cname):
    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            a = t.detach().float()
            rows.append((fname, cname, tuple(a.shape), a.abs().max().item()))
    return hook

for fname in ("init_repr_factory", "cond_factory", "pair_repr_builder"):
    f = getattr(net, fname, None)
    cs = getattr(f, "feat_creators", None)
    if cs is None:
        inner = getattr(f, "init_repr_factory", None) or getattr(f, "cond_factory", None)
        cs = getattr(inner, "feat_creators", []) if inner else []
    for c in cs:
        handles.append(c.register_forward_hook(mk(fname, type(c).__name__)))

with torch.no_grad():
    net(batch)
for h in handles: h.remove()

cur, live, dead = None, 0, 0
for fname, cname, shape, mx in rows:
    if fname != cur: print(f"\n=== {fname} ==="); cur = fname
    tag = "DEAD " if mx == 0.0 else "LIVE "
    if mx == 0.0: dead += 1
    else: live += 1
    print(f"  {tag}{cname:46} max {mx:.4g}")
print(f"\n  conditioned: live {live}, inert {dead}, total {live+dead}")
