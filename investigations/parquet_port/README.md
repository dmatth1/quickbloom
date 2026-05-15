# Parquet bloom port: AVX2 vs scalar Apache Arrow / Velox

Standalone microbench comparing the scalar Parquet bloom probe shipped
by Apache Arrow C++ / arrow-rs / Velox against an AVX2 probe on the
**same Parquet on-disk format**. Companion to `../PARQUET.md`.

## What this proves

1. **The Parquet ecosystem ships a scalar bloom probe** despite the
   spec being designed for AVX2 (256-bit blocks = one `__m256i`).
   Source: vendored line-for-line in `reference/`.
2. **A drop-in AVX2 probe is bit-identical to the scalar.** Verified
   over 167M (query, filter) pairs across three regimes — 0
   mismatches.
3. **Single-key probe is 3.4× faster in-cache, 1.15–1.23×
   out-of-L3.** A 4-way bulk probe across row-group filters widens
   the out-of-L3 win to 1.49–1.57× by overlapping cache-miss
   latency.
4. **Per-query CPU savings are significant on large-table scans**:
   ~224 µs/query on a 1 GiB-filter-footprint scan vs scalar.

See `RESULTS.md` for the numbers and `../PARQUET.md` for the broader
narrative including blast radius across the ecosystem.

## Layout

- `src/`
  - `xxh64.{h,cpp}` — minimal Parquet-spec XXH64 with self-test
    against canonical vectors.
  - `parquet_bloom.{h,cpp}` — three probe paths on the same
    Parquet-spec data structure:
    1. `find_scalar`: line-for-line Apache Arrow C++ `FindHash`.
    2. `find_avx2`: AVX2 probe, same on-disk layout.
    3. `find_avx2_x4`: 4-way bulk probe across row-group filters.
  - `bench.cpp` — workload generator + diff-test + 3-regime bench.
- `reference/` — verbatim source from upstream:
  - `arrow_cpp_bloom_filter.{h,cc}` from `apache/arrow`
  - `arrow_rs_bloom_filter.rs` from `apache/arrow-rs`
  - `velox_bloom_filter.cpp` from `facebookincubator/velox`
- `results/` — captured bench runs.

## Build & run

```
make && ./bench
```

Requires AVX2 + BMI2 (Haswell+/Zen2+).

## Mapping to upstream PRs

The benchmark's `find_avx2` body is a 6-line function. The same
6 lines drop into:

| Upstream | Function to replace |
|---|---|
| `apache/arrow` cpp/src/parquet/bloom_filter.cc | `BlockSplitBloomFilter::FindHash` |
| `apache/arrow-rs` parquet/src/bloom_filter/mod.rs | `Block::check` |
| `facebookincubator/velox` velox/dwio/parquet/common/BloomFilter.cpp | `BlockSplitBloomFilter::findHash` |

Behind a runtime AVX2 feature check (`cpuid` / `std::is_x86_feature_detected`)
with the scalar fallback retained for non-AVX2 builds. No format
change, no behavior change, no spec change.
