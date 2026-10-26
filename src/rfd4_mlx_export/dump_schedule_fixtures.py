#!/usr/bin/env python3
"""Dump upstream flow-matching schedules as parity fixtures for the Swift port.

The schedules are deterministic tensors, so the playbook wants them pinned
bitwise rather than compared with a tolerance. This writes upstream's exact
float32 values; `FlowScheduleParityTests.swift` reads them back.

Values are emitted as hex bit patterns of the float32s, not decimal, so the
comparison cannot be weakened by a round trip through a decimal literal.

Run with the upstream venv's python and -I:
    .venv/bin/python -I dump_schedule_fixtures.py <upstream-repo> <out.json>
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rfproteina.flow_matching.product_space_flow_matcher import (
    ProductSpaceFlowMatcher,
    get_schedule,
    get_schedule_tsr_safe,
)

DATA_MODES = ["bb_ca", "local_latents"]


class _ConfiguredFlowMatcher:
    """Just enough ProductSpaceFlowMatcher to build schedules.

    Mirrors the helper in upstream `tests/test_partial_schedule.py`.
    """

    data_modes = DATA_MODES
    sample_schedule = ProductSpaceFlowMatcher.sample_schedule


def bits(t: torch.Tensor) -> list[str]:
    """float32 values as 8-hex-digit big-endian bit patterns."""
    flat = t.detach().to(torch.float32).flatten().tolist()
    return [struct.pack(">f", v).hex() for v in flat]


def main() -> int:
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = OmegaConf.load(repo / "configs/generation/inference_base.yaml")

    fixtures: dict = {
        "torch": torch.__version__,
        "note": "float32 bit patterns, big-endian hex. Generated; do not hand-edit.",
        "released": {},
        "builders": {},
    }

    # 1. The released 60-step schedules, exactly as inference builds them.
    fm = _ConfiguredFlowMatcher()
    ts, gt = fm.sample_schedule(nsteps=60, sampling_model_args=cfg.model)
    for dm in DATA_MODES:
        fixtures["released"][dm] = {"ts": bits(ts[dm]), "gt": bits(gt[dm])}

    # 2. The raw builders, isolated from the config, across modes and tails.
    #    These separate "the schedule math is wrong" from "the config is wrong".
    for name, fn in (("get_schedule", get_schedule),
                     ("get_schedule_tsr_safe", get_schedule_tsr_safe)):
        for mode in ("uniform", "power", "log"):
            for tail_steps, mult in ((0, 1), (10, 2)):
                key = f"{name}/{mode}/tail{tail_steps}x{mult}"
                try:
                    v = fn(mode=mode, nsteps=60, p1=2.0,
                           tail_steps=tail_steps, tail_step_multiplier=mult)
                except Exception as exc:  # record refusals too -- they are contract
                    fixtures["builders"][key] = {"error": f"{type(exc).__name__}: {exc}"}
                    continue
                fixtures["builders"][key] = {"values": bits(v)}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixtures, indent=1) + "\n", encoding="utf-8")

    n = len(fixtures["builders"])
    print(f"wrote {out}")
    print(f"  released: {len(fixtures['released'])} modalities "
          f"(ts {len(fixtures['released']['bb_ca']['ts'])}, "
          f"gt {len(fixtures['released']['bb_ca']['gt'])})")
    print(f"  builders: {n} grids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
