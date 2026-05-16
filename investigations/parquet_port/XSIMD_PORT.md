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

## Comparison to the raw-intrinsics numbers

Earlier raw-`_mm256_*` numbers on Cascade Lake @ 2.8 GHz
(`results/run2.txt`) for reference:

| Regime             | scalar    | AVX2 single-key    | AVX2 4-way bulk    |
|--------------------|----------:|-------------------:|-------------------:|
| Small (in-cache)   |  12.73 ns |   3.70 (3.44×)     |   2.36 (5.40×)     |
| Medium (out-of-L3) |  35.84 ns |  29.18 (1.23×)     |  22.79 (1.57×)     |
| Large (deep DRAM)  |  41.41 ns |  35.90 (1.15×)     |  27.75 (1.49×)     |

Two things changed between the runs: the implementation (raw → xsimd)
and the host (Cascade Lake → Sapphire Rapids). Absolute numbers
shifted in both directions but the ratios are the same shape — xsimd
lowers to the same instruction sequence as the raw intrinsics on an
AVX2 build, so the implementation change alone shouldn't move the
ratio meaningfully on a single host. Cleanest A/B would be both
implementations on the same machine; not yet captured.

## What's not done

- **NEON port.** xsimd makes this mechanical (`xsimd::batch<uint32_t>`
  is 4-lane on NEON, so each 256-bit block needs two batches), but it
  needs measuring before claiming numbers.
- **`PARQUET.md`, `parquet_port/README.md`, `RESULTS.md` re-sync.**
  Those still describe the raw-intrinsics version and headline
  numbers. Deliberately left alone for now.
- **Blog post.** Same — still references the AVX2 intrinsics and the
  Cascade Lake numbers.
