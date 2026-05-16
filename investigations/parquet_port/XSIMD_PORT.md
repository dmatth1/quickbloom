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
| `_mm256_testc_si256(blk, mask)`       | `xsimd::all((blk & mask) == mask)`      |

`find_scalar` is unchanged — still the line-for-line port of
Apache Arrow C++'s `FindHash`, used as the diff-test reference.

The single-key probe body, after the rewrite:

```cpp
bool filter::find_simd(uint64_t hash) const {
    using batch_t = xsimd::batch<uint32_t>;  // 8-lane on AVX2 / AVX-512

    uint32_t bucket_index = uint32_t(
        ((hash >> 32) * (data_.size() / kBytesPerFilterBlock)) >> 32);
    uint32_t key = uint32_t(hash);

    batch_t hash_v = batch_t::broadcast(key);
    batch_t salt   = batch_t::load_aligned(SALT);
    batch_t prod   = hash_v * salt;
    batch_t shift  = prod >> 27;
    batch_t mask   = batch_t::broadcast(uint32_t(1)) << shift;

    batch_t blk = batch_t::load_unaligned(reinterpret_cast<const uint32_t*>(
        data_.data() + bucket_index * kBytesPerFilterBlock));
    return xsimd::all((blk & mask) == mask);
}
```

`static_assert(batch_t::size == 8)` is set in
`parquet_bloom.cpp` — Parquet's 8-bits-per-block expects an 8-lane
batch, which is the default on AVX2 and AVX-512. A NEON port would
need a different splitting (two 4-lane batches per block) and is
not yet implemented.

## Build

```
sudo apt install libxsimd-dev   # xsimd 12+
cd investigations/parquet_port
make && ./bench
```

xsimd is header-only; no link change. Same compile flags as before:
`-O3 -mavx2 -mbmi2 -mfma`. Build is clean under `-Wall -Wextra`.

## Bench

Captured run: `results/run3_xsimd.txt`. Hardware: **Intel Xeon
Sapphire Rapids @ 2.1 GHz (virtualised)** — different host from the
earlier Cascade Lake @ 2.8 GHz runs in `results/run1.txt` and
`results/run2.txt`.

All numbers below are post-hash probe latency: XXH64 is computed
once at setup and excluded from the timed loop. Median of 5
repeats.

| Regime             | Filter footprint | scalar    | xsimd single-key   | xsimd 4-way bulk   |
|--------------------|-----------------:|----------:|-------------------:|-------------------:|
| Small (in-cache)   |          0.5 MiB |  10.82 ns |  **2.50 (4.32×)**  |  **1.82 (5.95×)**  |
| Medium (out-of-L3) |          128 MiB |  27.16 ns |     23.46 (1.16×)  |  **15.28 (1.78×)** |
| Large (deep DRAM)  |            1 GiB |  38.41 ns |     31.02 (1.24×)  |  **26.28 (1.46×)** |

**Diff-test: 0 mismatches across 167M (query, filter) pairs** — the
xsimd path produces bit-identical results to the scalar Arrow C++ /
arrow-rs / Velox reference.

## Same-machine A/B: xsimd vs raw `_mm256_*`

Both implementations built and run on the same SPR host. Raw run
captured in `results/run4_raw_intrinsics_spr.txt`:

| Regime             | impl     | single-key ns/probe | 4-way ns/probe |
|--------------------|----------|--------------------:|---------------:|
| Small (in-cache)   | raw      |                2.22 |           1.57 |
|                    | xsimd    |                2.50 |           1.82 |
|                    | **xsimd vs raw** |    **+13%** |       **+16%** |
| Medium (out-of-L3) | raw      |               15.08 |          12.01 |
|                    | xsimd    |               23.46 |          15.28 |
|                    | **xsimd vs raw** |    **+56%** |       **+27%** |
| Large (deep DRAM)  | raw      |               30.29 |          25.72 |
|                    | xsimd    |               31.02 |          26.28 |
|                    | **xsimd vs raw** |     **+2%** |        **+2%** |

The xsimd path is consistently slower than the hand-tuned intrinsics.
Both are still much faster than scalar (xsimd is 4.3× vs scalar
in-cache; raw is 4.8×), and both diff-test bit-identical to the
scalar reference.

### Why xsimd is slower

`xsimd` has no direct equivalent of `_mm256_testc_si256` for our
use case. The "all mask bits set in block" test compiles to:

```
vpand    (mem), ymm_mask, ymm_tmp     ; tmp = blk & mask
vpcmpeqd ymm_tmp, ymm_mask, ymm_eq    ; per-lane (tmp == mask)
vpcmpeqd ymm_ones, ymm_ones, ymm_ones ; sentinel all-ones
vptest   ymm_ones, ymm_eq             ; CF = all lanes true
setb     %al
```

Five instructions vs the raw version's single `vptest` (which
computes `(~blk & mask) == 0` directly). On a 7-cycle hot path the
extra ops aren't free.

The out-of-L3 medium regime sees the biggest gap (+56%) because the
per-probe latency there is dominated by L3-miss-but-DRAM-pageHit
loads, and the extra ALU keeps the load port waiting longer. The
deep-DRAM regime narrows back to noise (+2%) because DRAM latency
dominates regardless.

### Trade-off

| Axis                          | Raw `_mm256_*`            | xsimd                                       |
|-------------------------------|---------------------------|---------------------------------------------|
| In-cache probe latency        | 2.22 ns                   | 2.50 ns (+13%)                              |
| Out-of-L3 probe latency       | 15.08 ns                  | 23.46 ns (+56%)                             |
| Deep-DRAM probe latency       | 30.29 ns                  | 31.02 ns (+2%)                              |
| Speedup over scalar (in-cache) | 4.81×                    | 4.32×                                       |
| Portability                   | x86 AVX2 only             | x86 AVX/AVX2/AVX-512 + ARM NEON/SVE2 + more |
| Upstream Arrow alignment      | No                        | Yes (Arrow already uses xsimd elsewhere)    |
| Diff-test vs scalar           | 0 mismatches / 167M       | 0 mismatches / 167M                         |

So the answer to "does xsimd affect anything other than compatibility"
is: yes, it costs measurable probe latency, especially out-of-L3.
The choice is: ship the abstraction Arrow already uses (xsimd) at
some perf cost, or ship hand-tuned intrinsics behind manual dispatch
(faster but a divergence from house style).

For the upstream pitch, xsimd is the right answer — the perf gap is
real but the path is still a clear win over scalar, and the
maintainer asked for it explicitly. Worth flagging in the PR
description so reviewers know the choice is informed.

### Older Cascade Lake numbers (different machine, raw intrinsics)

Captured earlier in `results/run2.txt` on Cascade Lake @ 2.8 GHz for
reference. Different hardware, raw-intrinsics implementation:

| Regime             | scalar   | AVX2 single-key | AVX2 4-way bulk |
|--------------------|---------:|----------------:|----------------:|
| Small (in-cache)   | 12.73 ns | 3.70 (3.44×)    | 2.36 (5.40×)    |
| Medium (out-of-L3) | 35.84 ns | 29.18 (1.23×)   | 22.79 (1.57×)   |
| Large (deep DRAM)  | 41.41 ns | 35.90 (1.15×)   | 27.75 (1.49×)   |

## What's not done

- **NEON port.** xsimd makes this mechanical (`xsimd::batch<uint32_t>`
  is 4-lane on NEON, so each 256-bit block needs two batches), but it
  needs measuring before claiming numbers.
- **`PARQUET.md`, `parquet_port/README.md`, `RESULTS.md` re-sync.**
  Those still describe the raw-intrinsics version and headline
  numbers. Deliberately left alone for now.
- **Blog post.** Same — still references the AVX2 intrinsics and the
  Cascade Lake numbers.
