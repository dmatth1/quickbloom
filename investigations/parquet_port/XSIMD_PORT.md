# Switching the parquet_port probe to xsimd

This document records the move from raw `_mm256_*` intrinsics to
[xsimd](https://github.com/xtensor-stack/xsimd). The existing
`PARQUET.md`, `parquet_port/README.md`, and `RESULTS.md` still
describe the raw-intrinsics version and have not been re-synced.

## Why

The `[DISCUSS]` reply on `dev@arrow.apache.org`
([thread](https://lists.apache.org/thread/omof0fq47tndfd80g5hwp2bvjmzvpb40))
asked that the probe be written against xsimd rather than raw
intrinsics, matching the rest of Arrow's SIMD code (e.g.
`arrow/util/bit_util.h`, `arrow/util/bpacking_simd.h`). For the
investigation to map cleanly to the eventual upstream patch, the
local code needs to use the same abstraction.

## What changed

Functions renamed and rewritten against `xsimd::batch<uint32_t>`:

| Before                                | After                                   |
|---------------------------------------|-----------------------------------------|
| `filter::find_avx2(uint64_t)`         | `filter::find_simd(uint64_t)`           |
| `filter::find_avx2_x4(f0..3, hash)`   | `filter::find_simd_x4(f0..3, hash)`     |
| `_mm256_*` intrinsics                 | `xsimd::batch<uint32_t>` ops            |
| `_mm256_testc_si256(blk, mask)`       | hybrid (see below)                      |

`find_scalar` is unchanged — still the line-for-line port of
Apache Arrow C++'s `FindHash`, used as the diff-test reference.

The mask compute is straight xsimd:

```cpp
inline batch_t simd_mask(uint32_t key) {
    batch_t hash_v = batch_t::broadcast(key);
    batch_t salt   = batch_t::load_aligned(SALT);
    batch_t prod   = hash_v * salt;
    batch_t shift  = prod >> 27;
    batch_t ones   = batch_t::broadcast(uint32_t(1));
    return ones << shift;
}
```

The "all mask bits set in block" reduction is the one place we reach
back to a raw intrinsic — see next section for why.

`static_assert(batch_t::size == 8)` is set in `parquet_bloom.cpp` —
Parquet's 8-bits-per-block expects an 8-lane batch, which is the
default on AVX2 and AVX-512. A NEON port would need a different
splitting (two 4-lane batches per block) and is not yet implemented.

## The reduction: an xsimd API gap

`_mm256_testc_si256(blk, mask)` returns 1 iff `(~blk & mask) == 0`,
which is exactly "every bit set in mask is also set in blk". That's
a single `vptest` instruction.

The natural xsimd translation is `xsimd::all((blk & mask) == mask)`.
On AVX2 that lowers to **five** instructions:

```
vpand    (mem), ymm_mask, ymm_tmp     ; tmp = blk & mask
vpcmpeqd ymm_tmp, ymm_mask, ymm_eq    ; per-lane (tmp == mask)
vpcmpeqd ymm_ones, ymm_ones, ymm_ones ; sentinel all-ones
vptest   ymm_ones, ymm_eq             ; CF = all lanes true
setb     %al
```

xsimd uses `_mm256_testc_si256` internally for `all(batch_bool)`
(see `xsimd_avx.hpp:138`) but doesn't expose
`xsimd::testc(batch, batch)` as a public primitive. Trying the
`bitwise_andnot` route lowers no better — `(bitwise_andnot(mask, blk) == 0)`
still needs the comparison-then-vptest pattern through the public
API.

The pragmatic fix: keep xsimd for the mask compute and the load,
reach for `_mm256_testc_si256` directly on AVX/AVX2 for the
reduction. One escape hatch, one `#if defined(__AVX__)`, with the
portable `xsimd::all(...)` form preserved as the fallback for other
arches:

```cpp
inline bool all_mask_bits_set(batch_t blk, batch_t mask) {
#if defined(__AVX__)
    return _mm256_testc_si256(__m256i(blk), __m256i(mask)) != 0;
#else
    return xsimd::all((blk & mask) == mask);
#endif
}
```

The right upstream fix is a small xsimd PR adding
`xsimd::testc(batch, batch)` as a portable primitive — `vptest` on
x86 AVX/AVX2/AVX-512, `vmaxvq + cmp` or similar on NEON, the natural
SVE2 reduction. That removes the escape hatch and gives other Arrow
code (or anyone else doing bloom probes / bitset intersection) the
same fast path.

## Build

```
sudo apt install libxsimd-dev   # xsimd 12+
cd investigations/parquet_port
make && ./bench
```

xsimd is header-only; no link change. Same compile flags as before:
`-O3 -mavx2 -mbmi2 -mfma`. Build is clean under `-Wall -Wextra`.

## Bench

20 repeats per regime, median reported alongside min / p90 / max for
variance. Captured run: `results/run5_xsimd_with_testc.txt`.

Hardware: **Intel Xeon Sapphire Rapids @ 2.1 GHz (virtualised)** —
different host from the earlier Cascade Lake @ 2.8 GHz runs in
`results/run1.txt` and `results/run2.txt`.

All numbers below are post-hash probe latency: XXH64 is computed
once at setup and excluded from the timed loop.

### xsimd-with-testc (this branch)

| Regime             |   min   | median  |   p90   |   max   |
|--------------------|--------:|--------:|--------:|--------:|
| Small (in-cache)   |         |         |         |         |
| · scalar           | 12.25   | 12.31   | 12.47   | 13.59   |
| · xsimd single-key |  2.29   |  2.33   |  2.41   |  2.70   |
| · xsimd 4-way bulk |  1.66   |  1.68   |  1.78   |  1.82   |
| Medium (out-of-L3) |         |         |         |         |
| · scalar           | 24.58   | 25.16   | 25.60   | 25.92   |
| · xsimd single-key | 15.12   | 16.61   | 17.44   | 17.83   |
| · xsimd 4-way bulk | 11.74   | 12.99   | 13.41   | 13.98   |
| Large (deep DRAM)  |         |         |         |         |
| · scalar           | 31.62   | 32.27   | 33.14   | 35.00   |
| · xsimd single-key | 23.28   | 23.65   | 24.46   | 24.77   |
| · xsimd 4-way bulk | 19.03   | 19.46   | 20.27   | 21.03   |

Speedups (median, vs scalar): **5.29×** / **7.33×** in-cache,
**1.51×** / **1.94×** out-of-L3 medium, **1.36×** / **1.66×** deep
DRAM.

**Diff-test: 0 mismatches across 167M (query, filter) pairs** — the
xsimd path produces bit-identical results to the scalar Arrow C++ /
arrow-rs / Velox reference.

## Same-machine A/B: xsimd vs raw `_mm256_*`

Both implementations built and run on the same SPR host with the
same 20-rep bench. Raw run captured in
`results/run6_raw_intrinsics_spr_20reps.txt`.

Median ns/probe:

| Regime             | impl  | single-key | 4-way bulk |
|--------------------|-------|-----------:|-----------:|
| Small (in-cache)   | raw   |      2.30  |      1.67  |
|                    | xsimd |      2.33  |      1.68  |
|                    | **xsimd vs raw** |  **+1%**   |  **+1%**   |
| Medium (out-of-L3) | raw   |     16.96  |     13.04  |
|                    | xsimd |     16.61  |     12.99  |
|                    | **xsimd vs raw** |  **−2%**   |  **~0%**   |
| Large (deep DRAM)  | raw   |     23.68  |     19.53  |
|                    | xsimd |     23.65  |     19.46  |
|                    | **xsimd vs raw** |  **~0%**   |  **~0%**   |

With the testc escape hatch the xsimd path is within noise of the
hand-tuned intrinsics across every regime. (An earlier measurement
showed +13% / +56% / +2% with the natural-but-pessimal
`xsimd::all((blk & mask) == mask)` reduction — and that was real,
not variance. The escape hatch closes it.)

## Note on the earlier "+56% out-of-L3" finding

A first version of this doc reported xsimd was 56% slower in the
medium-out-of-L3 regime with a hand-wave explanation about "extra
ALU keeping the load port waiting longer". On reflection that's
mechanically wrong — the load is in flight downstream of the ALU,
and extra ALU after the load doesn't extend the critical path.

The actual cause was the extra dependency chain on the reduction:
each iteration's `vpcmpeqd + vpcmpeqd(allones) + vptest` chain has
to retire before the next iteration's load address is computed (or
before the OoO core has window space to fetch it), so the prefetch
distance shrinks and effective MLP drops. Hard to confirm without
`perf stat -e l1d_pend_miss.pending` or similar, but it's a
plausible mechanism that the escape-hatch numbers are consistent
with.

## What's not done

- **NEON port.** xsimd makes the mask compute mechanical
  (`xsimd::batch<uint32_t>` is 4-lane on NEON, so each 256-bit block
  needs two batches). The `#if defined(__AVX__)` escape hatch on the
  reduction means the portable fallback would handle the test, but
  the algorithm split for an 8-lane Parquet block on a 4-lane batch
  still needs writing. Not done here.
- **`xsimd::testc(batch, batch)` upstream.** Would remove the escape
  hatch and benefit anyone else doing bitset intersection /
  all-bits-set tests. Small PR.
- **`PARQUET.md`, `parquet_port/README.md`, `RESULTS.md` re-sync.**
  Those still describe the raw-intrinsics version and headline
  numbers. Deliberately left alone for now.
- **Blog post.** Same — still references the AVX2 intrinsics and the
  Cascade Lake numbers.
