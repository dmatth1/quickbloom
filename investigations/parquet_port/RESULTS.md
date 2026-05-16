# Parquet bloom port: results

Measured on Intel Xeon @ 2.8 GHz (Cascade Lake-class, 33 MiB L3,
virtualised). Build: `g++ -O3 -mavx2 -mbmi2 -mfma`. Median of 5
repeats. All numbers are post-hash probe latency: XXH64 is computed
once at setup and excluded from the timed loop. **Diff-test: 0
mismatches across 167M probes** — the AVX2 path produces
bit-identical results to the scalar Arrow C++ / arrow-rs / Velox
reference.

## Headline numbers (ns/probe)

| Regime             | Total footprint | scalar (Arrow/Velox shape) | AVX2 single-key      | AVX2 4-way across row groups |
|--------------------|----------------:|---------------------------:|---------------------:|-----------------------------:|
| Small (in-cache)   |         0.5 MiB |                      12.73 |   **3.70 (3.44×)**   |        **2.36 (5.40×)**      |
| Medium (out-of-L3) |         128 MiB |                      35.84 |       29.18 (1.23×)  |        **22.79 (1.57×)**     |
| Large (deep DRAM)  |           1 GiB |                      41.41 |       35.90 (1.15×)  |        **27.75 (1.49×)**     |

## Per-query implication

For a `SELECT * WHERE col = 'X'` over an N-row-group table, every
query value probes every row group's bloom. Per-query CPU savings:

| Regime | N row groups | scalar query CPU | AVX2 4-way query CPU | Saved per query |
|--------|-------------:|-----------------:|---------------------:|----------------:|
| Small  |           64 |          0.81 µs |              0.15 µs |   0.66 µs (5.4×) |
| Medium |         4096 |        146.8 µs  |             93.4 µs  |     **53.5 µs**  |
| Large  |        16384 |        678.5 µs  |            454.6 µs  |      **224 µs**  |

## Why the in-cache gap is bigger than the out-of-L3 gap

Parquet SBBF was already cache-line-blocked in 2018 — one 256-bit
block (= half a cache line) per probe. That collapsed the
cache-miss-per-bit cost classical Bloom paid. The remaining cost is
the ALU loop computing 8 mask bits and AND-testing them against the
block.

That means:

- **In-cache**, where the ALU loop dominates: **3.4× single-key,
  5.4× with 4-way bulk**. The scalar 8-iteration loop with branches
  collapses to `vpmulld + vpsrld + vpsllvd + vptestc`. Real
  ecosystem win because warm-filter queries do hit in cache.
- **Out-of-L3**, where memory latency dominates: **1.15–1.23×
  single-key, 1.49–1.57× with 4-way bulk**. Both paths wait for the
  same DRAM load, so the per-probe gain narrows. The 4-way bulk
  recovers throughput by overlapping four cache-miss latencies in
  flight against four different row-group filters.

## What this means for an upstream PR

A drop-in AVX2 `findHash` replacement in `arrow-cpp` / `arrow-rs` /
`velox` ships ~3× per-probe speedup for the in-cache hot path (think
analyst running thousands of small point queries against a hot
table) and ~1.2× for the cold scan case. That's worth a PR by
itself because the diff-test verifies bit-identical output — no
behaviour change for any reader.

The **bigger** win is exposing a bulk-probe API that overlaps row-
group probes within a query. That's a small API addition but lets
the engine call `find_avx2_x4(filters[i..i+4], hash)` to fetch four
cache lines in parallel — 1.5× out-of-L3 for free. That's where
the cold-scan large-table workload (the one Iceberg/Hudi/Delta
users hit most often) actually feels the difference.

## Honest limitations

- **Virtualised hardware.** Bare-metal Xeon should see slightly
  wider gaps because the hypervisor doesn't perfectly mirror cache
  behaviour. ARM equivalents (Graviton) and AMD (Zen 3+) would
  need separate measurements; the AVX2 path translates to NEON /
  Zen predictably but the gap might shift.
- **No prefetch path measured.** Wide OoO cores hide a lot of
  latency natively; explicit prefetch may or may not pay back for
  row-group filters scattered across DRAM.
- **Single-thread.** Real engines parallelise across row groups
  with multiple threads. Per-thread per-probe numbers translate
  but the aggregate cluster effect depends on memory bandwidth
  contention.
- **No XXH64 SIMD path.** Parquet mandates XXH64; computing it
  faster via SIMD for batched queries (`IN (...)`) is a separate
  optimisation that compounds with the AVX2 probe speedup but
  isn't measured here.

## Reproducing

```
cd investigations/parquet_port
make && ./bench
```

Requires AVX2 + BMI2 (Haswell+/Zen2+). Builds with gcc 13 or clang 17.
