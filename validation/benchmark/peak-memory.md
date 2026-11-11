# Peak GPU memory, and where the 638-token regression came from

Measured 2026-11-11 on an M4, uncontended, with `BatchBench`, which builds the
real `TrunkLayer` and `PairReprUpdate` against a synthetic weight dictionary of
the checkpoint's shapes — so what is reported is the production forward pass, not
a replica of it. One trunk pass, `MLX.GPU.resetPeakMemory()` after a warm pass.

Reproduce with `tools/memory-walk.sh`, or a single point with
`RFD4_DTYPE=fp16 BMODE=mem BN=638 BB=1 ./.build/release/BatchBench`.

## Which commit did it

Walking `Sources/RFD4MLX` through each optimization commit while keeping the
benchmark at HEAD, at 638 tokens, fp32:

| commit | peak | Δ | pass |
|---|---|---|---|
| `68659fd` before all three | 8.866 GB | — | 10.25 s |
| `0a1674e` fused layerNorm/rmsNorm | 11.780 GB | **+2.914** | 8.03 s |
| `01040cf` pre-transposed pair weights | 12.198 GB | +0.418 | 7.64 s |
| `8827d53` MLX.compile elementwise fusion | 13.240 GB | +1.042 | 6.80 s |

**Two thirds of the regression is `0a1674e`**, the fused norm kernels — which is
also where most of the speed came from (1.28x on its own; 1.51x for all three).
The trade is real and it is not an accident: the hand-rolled layerNorm made seven
elementwise passes over a tensor that is 0.58 GB at n=638 but allocated little;
`MLXFast.layerNorm` makes one pass and more room.

`01040cf`'s +0.418 GB is the one piece that is live data rather than transient —
it stores a second, contiguous, pre-transposed copy of the pair-track weights,
and `active` memory steps by the same amount at that commit.

## It is not reclaimable cache

The obvious hypothesis — that `peak` counts the allocator's cache and the cache
simply grew — is **wrong**, and worth recording so nobody re-tests it:

| cache cap | peak | residual cache |
|---|---|---|
| none | 13.240 GB | 14.85 GB |
| 8 GB | 13.240 GB | 8.34 GB |
| 4 GB | 13.240 GB | 4.17 GB |
| 2 GB | 13.240 GB | 2.51 GB |
| 1 GB | 13.240 GB | 1.25 GB |

Capping the cache changes what is held afterwards and does not move the peak at
all. The allocator genuinely touches 13.24 GB at some instant during the pass, so
this is a real high-water mark and a real constraint on what sizes can be offered.
(`active` — live data — is only 1.77 GB, so the gap is transient working buffers,
not retained state. It is still memory that has to exist at that moment.)

## The curve

Peak is quadratic in token count, as the pair track implies:

```
fp32   peak(n) = 3.12637e-05 n^2 - 1.62935e-03 n + 1.559   GB   (max fit error 16 MB)
fp16   peak(n) = 1.56318e-05 n^2 - 8.14674e-04 n + 0.780   GB   (max fit error  8 MB)
```

| n | fp32 | fp16 |
|---|---|---|
| 256 | 3.196 GB | 1.598 GB |
| 384 | 5.528 GB | 2.764 GB |
| 512 | 8.936 GB | 4.468 GB |
| 638 | 13.240 GB | 6.620 GB |
| 768 | 18.75 GB (extrapolated) | 9.37 GB (extrapolated) |

fp16 is **exactly 2.000x smaller at every size measured**. That ratio is itself a
check on the dtype fix: any tensor still left in fp32 inside the trunk would pull
it below 2.

## What to do about it

Nothing, on this evidence, beyond shipping fp16 — which halves it and is the
default. The regression buys 1.51x of wall clock, and at fp16 the 638-token peak
is 6.62 GB rather than 13.24.

If a size guard is needed (and it is, where exhausting GPU memory aborts the host
process rather than throwing), the fp16 curve above is what to build it from, not
the fp32 one.

Reverting `0a1674e` is the lever if memory ever matters more than speed: it is
worth 2.9 GB at fp32 / 1.5 GB at fp16, against 1.28x of pass time. Reverting
`8827d53` is a second 1.0 GB / 0.5 GB for less speed. Neither is worth doing
pre-emptively.
