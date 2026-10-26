#!/usr/bin/env python3
"""Dump atom_site columns as biotite reads them, as a reference for the Swift reader.

The Swift reader targets the raw atom_site loop, so the right reference is a raw
CIF parse rather than a structure-level one: biotite's CIFFile gives the columns
exactly as written, before any convention is applied on top.

Usage: python dump_cif_reference.py <out.json> <name=path> [<name=path> ...]
"""
import json, sys
from biotite.structure.io.pdbx import CIFFile

def main():
    out, specs = sys.argv[1], sys.argv[2:]
    ref = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        f = CIFFile.read(path)
        block = f.block
        site = block["atom_site"]
        cols = {}
        for key in ("label_comp_id", "label_atom_id", "type_symbol",
                    "label_asym_id", "label_seq_id",
                    "Cartn_x", "Cartn_y", "Cartn_z"):
            cols[key] = [str(v) for v in site[key].as_array(str)]
        ref[name] = {"n_atoms": len(cols["label_comp_id"]), "columns": cols}
        print(f"  {name}: {ref[name]['n_atoms']} atoms, "
              f"{len(set(cols['label_asym_id']))} chains, "
              f"{len(set(cols['label_comp_id']))} distinct comp_ids")
    json.dump({"note": "biotite CIFFile raw atom_site columns. Generated.", "files": ref},
              open(out, "w"), indent=1)
    print(f"wrote {out}")
    return 0

raise SystemExit(main())
