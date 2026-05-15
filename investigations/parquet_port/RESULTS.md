# Parquet bloom port: results

Measured on Intel Xeon @ 2.8 GHz (Cascade Lake-class, 33 MiB L3,
virtualised). Build: `g++ -O3 -mavx2 -mbmi2 -mfma`. Median of 5
repeats. **Diff-test: 0 mismatches across 167M probes** — the AVX2
path produces bit-identical results to the scalar Arrow C++ /
arrow-rs / Velox reference.

## Headline numbers (ns/probe)

| Regime | Total footprint | scalar (Arrow/Velox shape) | AVX2 single-key (v14) | AVX2 4-way across row groups (v11) |
|---|---:|---:|---:|---:|
| Small (in-cache) | 0.5 MiB | 12.73 | **3.70 (3.44×)** | **2.36 (5.40×)** |
| Medium (out-of-L3) | 128 MiB | 35.84 | 29.18 (1.23×) | **22.79 (1.57×)** |
| Large (deep DRAM) | 1 GiB | 41.41 | 35.90 (1.15×) | **27.75 (1.49×)** |

## Per-query implication

For a `SELECT * WHERE col = 'X'` over an N-row-group table, every
query value probes every row group's bloom. Per-query CPU savings:

| Regime | N row groups | scalar query CPU | AVX2 4-way query CPU | Saved per query |
|---|---:|---:|---:|---:|
| Small | 64 | 0.81 µs | 0.15 µs | 0.66 µs (5.4×) |
| Medium | 4096 | 146.8 µs | 93.4 µs | **53.5 µs** |
| Large | 16384 | 678.5 µs | 454.6 µs | **224 µs** |

## What this tells us about the v14 win in Parquet

The story is **different from Scylla**, and it's worth understanding why:

- **Scylla** used classical Bloom — K probes to K *different* cache
  lines. The v14 win there was both ALU work and cache locality
  (collapsing K cache misses to 1). 2× on the bloom slice of a Get.
- **Parquet** already uses SBBF — one 256-bit block (= half a cache
  line) per probe. The format already won the cache-locality argument
  in 2018. The v14 win here is *only* the ALU work — replacing the
  scalar 8-iteration loop with `vpmulld + vpsrld + vpsllvd + vptestc`.

That means:

- **In-cache**, where ALU dominates: **3.4× single-key, 5.4× with
  4-way bulk**. Real ecosystem win because warm-filter queries do hit
  in cache.
- **Out-of-L3**, where memory latency dominates: **1.15–1.23×
  single-key, 1.49–1.57× with 4-way bulk**. Smaller per-probe gain
  because both paths wait for the same DRAM load — but the 4-way
  unroll moves the needle by overlapping cache-miss latency across
  row-group filters.

The 4-way bulk path is the v11 `bulk_is_present` pattern adapted to
probe one query hash against 4 row-group filters in parallel. The OoO
core can have 4 cache misses in flight at once, which is what
recovers throughput in the memory-bound regime.

## What this means for an upstream PR

A drop-in AVX2 `findHash` replacement in `arrow-cpp` / `arrow-rs` /
`velox` ships ~3× per-probe speedup for the in-cache hot path (think
analyst running thousands of small point queries against a hot
table) and ~1.2× for the cold scan case. That's worth a PR by itself
because the diff-test verifies bit-identical output — no behavioral
change for any reader.

The **bigger** win is exposing a bulk-probe API that overlaps row-
group probes within a query. That's a small API addition but lets
the engine call `find_avx2_x4(filters[i..i+4], hash)` to fetch four
cache lines in parallel — 1.5× out-of-L3 for free. That's where
the cold-scan large-table workload (the one Iceberg/Hudi/Delta users
hit most often) actually feels the difference.

## Honest limitations

- **Virtualised hardware.** Bare-metal Xeon should see slightly
  wider gaps because the hypervisor doesn't perfectly mirror cache
  behavior. ARM equivalents (Graviton) and AMD (Zen 3+) would need
  separate measurements; the AVX2 path translates to NEON / Zen
  predictably but the gap might shift.
- **No prefetch path measured.** Explicit prefetch was a wash in the
  original v11 work on this hardware (wide OoO core hides latency
  natively). For row-group filters scattered across DRAM, prefetch
  might help; not tested here.
- **Single-thread.** Real engines parallelise across row groups
  with multiple threads. Per-thread per-probe numbers translate
  but the aggregate cluster effect depends on memory bandwidth
  contention.
- **No XXH64 SIMD path.** Parquet mandates XXH64; computing it
  faster via SIMD for batched queries (`IN (...)`) is a separate
  optimisation that compounds with v14's probe speedup but isn't
  measured here.

## Reproducing

```
cd investigations/parquet_port
make && ./bench
```

Requires AVX2 + BMI2 (Haswell+/Zen2+). Builds with gcc 13 or clang 17.
