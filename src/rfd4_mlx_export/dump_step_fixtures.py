#!/usr/bin/env python3
"""Dump simulation_step (input, output) pairs across the branches it can take.

The step is a branch on t: deterministic above t_lim_ode and below
t_lim_ode_below, stochastic in between. A fixture at one t exercises one branch,
so this sweeps t over all three regions for both modalities. Noise is supplied
explicitly rather than drawn inside, so the stochastic branch is reproducible.

Usage: python dump_step_fixtures.py <out.json>
"""
import base64, json, sys
import torch
from omegaconf import OmegaConf
from rfproteina.flow_matching.rdn_flow_matcher import RDNFlowMatcher

def b64(t): return base64.b64encode(t.detach().contiguous().to(torch.float32).numpy().tobytes()).decode()
def entry(t): return {"shape": list(t.shape), "b64": b64(t)}

def main():
    out = sys.argv[1]
    g = torch.Generator().manual_seed(7)
    cases = []
    # (mode, dim, tsr_k, tsr_sigma)
    for mode, dim, k, sig in (("bb_ca", 3, 16.0, 3.0), ("local_latents", 32, 4.0, 12.0)):
        for t_val in (0.005, 0.5, 0.98):        # below / stochastic / above
            n = 8
            x_t = torch.randn(1, n, dim, generator=g)
            v = torch.randn(1, n, dim, generator=g)
            noise = torch.randn(1, n, dim, generator=g)
            emit = torch.ones(1, n, dtype=torch.bool)
            t = torch.full((1,), t_val)
            dt, gt = 0.01, 0.37

            # Patch randn so the stochastic branch consumes OUR noise.
            real_randn = torch.randn
            torch.randn = lambda *a, **kw: noise if (a and a[0] == noise.shape) else real_randn(*a, **kw)
            try:
                class _FM:
                    guidance_enabled = False
                    _apply_emit_mask = RDNFlowMatcher._apply_emit_mask
                    _force_zero_com = RDNFlowMatcher._force_zero_com
                x_next = RDNFlowMatcher.simulation_step(
                    _FM(), x_t=x_t, nn_out={"v": v}, t=t, dt=dt, gt=gt,
                    emit_mask=emit,
                    simulation_step_params={
                        "sampling_mode": "vf_tsr", "sc_scale_noise": 1.0,
                        "t_lim_ode": 0.95, "t_lim_ode_below": 0.02,
                        "center_every_step": False, "tsr_k": k, "tsr_sigma": sig,
                    },
                )
            finally:
                torch.randn = real_randn
            cases.append({
                "mode": mode, "t": t_val, "dt": dt, "gt": gt,
                "tsr_k": k, "tsr_sigma": sig,
                "x_t": entry(x_t), "v": entry(v), "noise": entry(noise),
                "emit": entry(emit.to(torch.float32)), "x_next": entry(x_next),
            })
    json.dump({"note": "base64 little-endian float32. Generated.",
               "torch": torch.__version__, "cases": cases}, open(out, "w"), indent=1)
    print(f"wrote {out} with {len(cases)} cases")
    for c in cases:
        print(f"  {c['mode']:14} t={c['t']:<6} -> |x_next| max {float(torch.as_tensor([0.0])):.0f}"
              f" shape {tuple(c['x_next']['shape'])}")
    return 0

raise SystemExit(main())
