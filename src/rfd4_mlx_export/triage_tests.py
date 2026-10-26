#!/usr/bin/env python3
"""Triage the RFD4-Proteina pytest suite into a Swift/MLX port manifest.

Reads test sources as TEXT only -- nothing is imported or executed.
Usage: python3 -I triage_tests.py <path-to-RFD4-Proteina> <out.md>
"""
import ast
import re
import sys
from pathlib import Path

# Upstream concerns that have no counterpart in a Swift/MLX inference port.
UNPORTABLE = {
    "slurm": "SLURM job submission - no meaning on device",
    "gradient": "backward pass - inference-only port has none",
    "cueq": "NVIDIA fused kernel vs fallback - fallback IS the reference",
    "entrypoint": "Hydra config resolution - no Hydra in Swift",
    "oracle": "needs AF2/AF3/OpenFold3 as an external oracle",
    "af2_acceptance": "needs AlphaFold2 - no AF2 port exists",
    "af3_structure_quality": "needs AF3/OpenFold3 baselines",
    "gpu_regression": "CUDA regression baselines",
    "torch_compile": "torch.compile plumbing - MLX lazy graph replaces it",
    "ema": "training-time EMA callback",
    "sweep_and_scripts": "repo tooling / shell scripts",
    "setup_search_envs": "conda env provisioning",
}
# Directories whose whole contents are out of scope for an inference port.
UNPORTABLE_DIRS = {"callbacks": "Lightning training callbacks",
                   "metrics": "eval suite, needs oracle environments"}

DEP_PATTERNS = {
    "atomworks": re.compile(r"\batomworks\b"),
    "biotite": re.compile(r"\bbiotite\b"),
    "scipy": re.compile(r"\bscipy\b"),
    "cuda": re.compile(r"\b(cuda|is_available|\.gpu)\b"),
    "checkpoint": re.compile(r"from_pretrained|hf_hub|safetensors"),
    "torch": re.compile(r"\btorch\b"),
}


def classify(rel: Path, text: str) -> tuple[str, str]:
    top = rel.parts[0] if len(rel.parts) > 1 else ""
    if top in UNPORTABLE_DIRS:
        return "DROP", UNPORTABLE_DIRS[top]
    stem = rel.stem.replace("test_", "")
    for key, why in UNPORTABLE.items():
        if key in stem:
            return "DROP", why
    if "@pytest.mark.gpu" in text and "atomworks" not in text:
        return "REVIEW", "gpu-marked; check whether the assertion is device-independent"
    return "PORT", ""


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2])
    tests = root / "tests"
    rows, totals = [], {"PORT": 0, "DROP": 0, "REVIEW": 0}
    fn_totals = {"PORT": 0, "DROP": 0, "REVIEW": 0}

    for path in sorted(tests.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(tests)
        try:
            tree = ast.parse(text)
            fns = [n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name.startswith("test_")]
        except SyntaxError:
            fns = re.findall(r"^\s*def (test_\w+)", text, re.M)
        deps = sorted(k for k, p in DEP_PATTERNS.items() if p.search(text))
        verdict, why = classify(rel, text)
        totals[verdict] += 1
        fn_totals[verdict] += len(fns)
        rows.append({"file": str(rel), "loc": text.count("\n") + 1, "fns": fns,
                     "deps": deps, "verdict": verdict, "why": why})

    with out.open("w", encoding="utf-8") as f:
        f.write("# RFD4-Proteina test suite -> Swift/MLX port manifest\n\n")
        f.write(f"Generated from `{root}`. Sources read as text; nothing imported.\n\n")
        f.write(f"**{sum(len(r['fns']) for r in rows)} test functions "
                f"across {len(rows)} files.**\n\n")
        f.write("| verdict | files | test fns |\n|---|---|---|\n")
        for v in ("PORT", "REVIEW", "DROP"):
            f.write(f"| {v} | {totals[v]} | {fn_totals[v]} |\n")

        for v, title in (("PORT", "Port these"),
                         ("REVIEW", "Need a decision"),
                         ("DROP", "Not portable")):
            f.write(f"\n## {title}\n\n")
            f.write("| file | LOC | tests | deps | note |\n|---|---|---|---|---|\n")
            for r in (x for x in rows if x["verdict"] == v):
                f.write(f"| `{r['file']}` | {r['loc']} | {len(r['fns'])} | "
                        f"{', '.join(r['deps']) or '-'} | {r['why'] or ''} |\n")

        f.write("\n## Full test inventory\n\n")
        for r in rows:
            if r["verdict"] == "DROP":
                continue
            f.write(f"\n### `{r['file']}` — {r['verdict']} ({len(r['fns'])} tests)\n\n")
            for fn in r["fns"]:
                f.write(f"- [ ] `{fn}`\n")

    print(f"{sum(len(r['fns']) for r in rows)} test fns, {len(rows)} files")
    for v in ("PORT", "REVIEW", "DROP"):
        print(f"  {v:7} {totals[v]:3} files  {fn_totals[v]:4} tests")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
