#!/usr/bin/env python3
"""Emit the RFD4 147-class residue vocabulary as a Swift-loadable JSON resource.

The upstream YAML is the source of truth and is part of the weights:

    "Checkpoints depend on the residue order and each residue's atom-slot order.
     Reordering either changes what the learned weights mean."
     -- resources/aa_rna_dna_glycan_ncaa.yaml:3

So this does not reinterpret anything. It transcribes the YAML into JSON,
preserving group order and within-group order, pads each layout to 37 slots,
and re-runs upstream's own validation (`rfproteina/encoding.py:_load_vocabulary`)
so a bad transcription fails here rather than in Swift.

Parsed with a line scanner rather than PyYAML: the file is a flat two-level
mapping of quoted scalars, and this keeps the exporter dependency-free so it can
run under `python3 -I` on a machine with no upstream environment installed.

Usage:
    python3 -I gen_vocabulary.py <aa_rna_dna_glycan_ncaa.yaml> <vocabulary.json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

N_SLOTS = 37

# rfproteina/encoding.py:11. Note "mask" and "unknown" deliberately share id 4.
TOKEN_TYPE_IDS = {
    "aa": 0, "rna": 1, "dna": 2, "atomized": 3,
    "unknown": 4, "glycan": 5, "ncaa": 6, "mask": 4,
}

_GROUP = re.compile(r'^  ([a-z_]+):\s*$')
_ENTRY = re.compile(r'^    "(?P<name>[^"]+)":\s*"(?P<layout>[^"]*)"\s*$')
_SCALAR = re.compile(r'^(?P<key>[a-z_]+):\s*"?(?P<value>[^"#]*?)"?\s*(?:#.*)?$')


def parse(text: str) -> tuple[dict[str, str], list[tuple[str, list[tuple[str, list[str]]]]]]:
    """Return (top-level scalars, ordered [(group, [(residue, 37 slots)])])."""
    meta: dict[str, str] = {}
    groups: list[tuple[str, list[tuple[str, list[str]]]]] = []
    in_residues = False

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line == "residues:":
            in_residues = True
            continue
        if not in_residues:
            if m := _SCALAR.match(line):
                meta[m.group("key")] = m.group("value").strip()
            continue
        if m := _GROUP.match(line):
            name = m.group(1)
            if name not in TOKEN_TYPE_IDS:
                raise ValueError(f"line {lineno}: unknown token type {name!r}")
            groups.append((name, []))
            continue
        if m := _ENTRY.match(line):
            if not groups:
                raise ValueError(f"line {lineno}: residue outside any group")
            atoms = ["" if a == "-" else a for a in m.group("layout").split()]
            # Upstream validation, verbatim in intent (encoding.py:23-26).
            occupied = [a for a in atoms if a]
            if not 1 <= len(atoms) <= N_SLOTS:
                raise ValueError(f"line {lineno}: {m.group('name')!r} has {len(atoms)} slots")
            if len(set(occupied)) != len(occupied):
                raise ValueError(f"line {lineno}: {m.group('name')!r} repeats an atom name")
            groups[-1][1].append((m.group("name"), atoms + [""] * (N_SLOTS - len(atoms))))
            continue
        raise ValueError(f"line {lineno}: unparsed: {raw!r}")

    return meta, groups


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    meta, groups = parse(src.read_text(encoding="utf-8"))

    classes = [(g, r, slots) for g, residues in groups for r, slots in residues]
    expected = int(meta.get("expected_num_classes", 0))
    if len(classes) != expected:
        raise SystemExit(f"got {len(classes)} classes, YAML declares {expected}")
    names = [r for _, r, _ in classes]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate residue name across groups")

    payload = {
        "name": meta.get("name", src.stem),
        "ccdSnapshot": meta.get("ccd_snapshot", ""),
        "slots": N_SLOTS,
        "classes": [
            {"name": r, "group": g, "tokenTypeId": TOKEN_TYPE_IDS[g], "atoms": slots}
            for g, r, slots in classes
        ],
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    counts = {g: len(r) for g, r in groups}
    print(f"{len(classes)} classes -> {dst}")
    print("  " + "  ".join(f"{g}={n}" for g, n in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
