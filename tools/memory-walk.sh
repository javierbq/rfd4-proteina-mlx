#!/bin/bash
# Walk Sources/RFD4MLX through the optimization commits and measure peak GPU
# memory at each, to find which one roughly doubled it at 638 tokens
# (6.69 -> 11.93 GB).
#
# The library directory is swapped to each historical state IN PLACE while the
# benchmark and the manifest stay at HEAD. Three reasons that beats the obvious
# alternatives:
#
#   - `git revert` of these commits conflicts: 01040cf and 8827d53 both rewrite
#     PairUpdate.swift, and 0a1674e also touches a Package.swift and a bench
#     target that no longer exist.
#   - A worktree per commit needs its own .build, and compiling MLX from scratch
#     four times costs more than the measurement.
#   - BatchBench does not exist at those commits, so the instrument has to come
#     from HEAD regardless. It only uses the public constructors, which these
#     commits do not change.
#
# `rm -rf` before each checkout matters: `git checkout <sha> -- <dir>` only
# writes the files present in that tree, so a file added later would otherwise
# survive into a checkout that predates it and fail to compile.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${N:-638}
B=${B:-1}
OUT=${OUT:-/tmp/rfd4-memory-walk.txt}

# Oldest first, so the report reads as a timeline. 0a1674e^ is the state before
# any of the three landed -- without it there is no baseline to call a doubling.
COMMITS=(
  "0a1674e^:before all three"
  "0a1674e:fused layerNorm and rmsNorm kernels"
  "01040cf:pre-transposed pair-track weights"
  "8827d53:MLX.compile elementwise fusion"
  "HEAD:HEAD"
)

restore() {
  rm -rf Sources/RFD4MLX
  git checkout HEAD -- Sources/RFD4MLX
  git checkout HEAD -- Package.swift
  echo "restored working tree to HEAD"
}
trap restore EXIT

# RFD4Export and RFD4Pack postdate every commit measured here and will not
# compile against the older library, so they sit out the walk.
python3 -I - <<'PY'
import pathlib, re
p = pathlib.Path("Package.swift")
s = p.read_text()
s = re.sub(r'        // Builds the installable weight pack\..*?\n        \),\n', '', s, flags=re.S)
p.write_text(s)
PY

: > "$OUT"
for entry in "${COMMITS[@]}"; do
  sha=${entry%%:*}
  label=${entry#*:}
  echo "=== $sha  ($label)"
  rm -rf Sources/RFD4MLX
  git checkout "$sha" -- Sources/RFD4MLX
  rm -f Sources/RFD4MLX/RFD4Pack.swift
  if ! swift build -c release --product BatchBench > /tmp/rfd4-memwalk-build.log 2>&1; then
    echo "  BUILD FAILED (see /tmp/rfd4-memwalk-build.log)" | tee -a "$OUT"
    tail -5 /tmp/rfd4-memwalk-build.log
    continue
  fi
  line=$(BMODE=mem BN=$N BB=$B ./.build/release/BatchBench 2>&1 | grep "^MEM" || true)
  resolved=$(git rev-parse --short "$sha")
  printf '%-10s %-42s %s\n' "$resolved" "$label" "$line" | tee -a "$OUT"
done

echo
echo "--- summary ($N tokens, batch $B)"
cat "$OUT"
