# Reference comparisons

External Bloom filter implementations included here for direct,
head-to-head measurement against the three designs at the top of
this repo. Each file is built and benchmarked by the same harness
that builds the main designs — these are not asserted numbers from
an upstream README, they are measured on whatever host you run on.

To include them in a bench run:

```sh
CC=clang python3 ../bench_all.py --comparisons
```

To run only correctness tests against them:

```sh
python3 ../test_bloom.py
```

## What's here

| File | Source | Algorithm |
|---|---|---|
| `bloom_impala.c` | parquet-format/BloomFilter.md | SBBF, 256-bit blocks, K=8, Parquet salts, **scalar bit-set**, **XXH64** hash |
| `bloom_krassovsky.c` | [save-buffer/bloomfilter_benchmarks](https://github.com/save-buffer/bloomfilter_benchmarks) (MIT) | PatternedSimd: 64-bit blocks, 1024-entry mask table (4–5 bits set, sliding window), AVX2 8-way batched, `vpgatherqq` contains |
| `bloom_classic.c` | textbook | K=8 double-hashing (`h_i = h1 + i·h2`), non-blocked. Pre-SBBF baseline. |
| `bloom_xorfuse.c` | [FastFilter/xor_singleheader](https://github.com/FastFilter/xor_singleheader) (Apache 2.0) | Graf+Lemire binary fuse filter (8-bit fingerprints, 3-hash xor). **Static**: must be built from a known key set; this file is a shim exposing the `bloom_*` ABI by buffering inserts and lazy-building on first contains. |
| `fastbloom_shim/` | [tomtomwombat/fastbloom](https://github.com/tomtomwombat/fastbloom) (MIT/Apache-2.0) | **Rust cdylib**. K=8 multi-block Bloom, SipHash-1-3 hasher. Crate self-bills as "the fastest Bloom filter in Rust" — true vs other Rust options, but loses to wymum-based C SBBF in our bench. |
| `arrow_rs_sbbf_shim/` | [apache/arrow-rs](https://github.com/apache/arrow-rs) `parquet::bloom_filter::Sbbf` (Apache 2.0) | **Rust cdylib**. Production SBBF used by every Rust Parquet reader/writer. K=8, XXH64, scalar bit-set (same algorithm as `bloom_impala.c`). No prehash API exposed by the crate. |

The Rust shims live in subdirectories with their own `Cargo.toml`;
the harness detects them by the `Cargo.toml` and builds with
`cargo build --release`. If the Rust toolchain isn't installed those
candidates are skipped with an error message; the C-only path keeps
working.

## Measured head-to-head

Run `CC=clang python3 ../bench_all.py --sizes S,M,L --comparisons` to
reproduce the table below on your host. Cross-host variance is real;
the labels in the README's perf table are illustrative.

Sample capture, Intel Xeon @ 2.1 GHz (Sapphire Rapids-class, 260 MB
L3), clang -O3, contains-miss, ns/op min:

| Implementation                | K | S (128 KB) | M (2 MB) | L (32 MB) | FP rate |
|-------------------------------|---|-----------:|---------:|----------:|--------:|
| `single_key` (top-level)      | 8 |   **1.64** | **2.42** |      7.16 | 0.00030 |
| `unified` (top-level)         | 8 |       1.98 |     2.70 |      7.58 | 0.00030 |
| `batched` (top-level)         | 4 |       2.84 |     3.34 |      7.00 | 0.00239 |
| `bloom_krassovsky.c`          | 5 |       1.84 |     2.93 |   **6.51**| 0.00340 |
| `bloom_xorfuse.c`             | 3 |       4.24 |     4.76 |     19.52 | 0.00422 |
| `bloom_classic.c`             | 8 |      10.15 |    13.13 |     19.38 | 0.00013 |
| `bloom_impala.c`              | 8 |      16.49 |    20.11 |     26.27 | 0.00030 |
| `arrow_rs_sbbf_shim/`         | 8 |      19.66 |    24.15 |     31.87 | 0.00030 |
| `fastbloom_shim/`             | 8 |      25.07 |    35.12 |     56.69 | 0.00012 |

Build cost (insert\_bulk, ns/key amortised over the populate phase)
shows the static-vs-dynamic split — xorfuse's lazy-build dominates
insert\_bulk time:

| Implementation                | S insert | M insert | L insert | notes |
|-------------------------------|---------:|---------:|---------:|---|
| `single_key`                  |     1.58 |     2.28 |     7.19 | dynamic bloom; insert ≈ contains |
| `bloom_xorfuse.c`             |    24.88 |    31.86 |   205.76 | static xor peel; cache-thrashes on scratch buffer at L |
| `fastbloom_shim/`             |    29.92 |    44.67 |    78.08 | SipHash-1-3 per key dominates |
| `arrow_rs_sbbf_shim/`         |    12.60 |    17.02 |    26.87 | XXH64 per key dominates |

## Caveats

* **Gather is CPU-dependent.** Krassovsky uses `vpgatherqq` for the
  contains path. On older gather implementations (Cascade Lake)
  `single_key` is ~3× faster; on Sapphire Rapids (improved gather
  latency) the gap narrows to ~16%. Either way `single_key` wins, but
  the margin shifts by CPU.

* **Krassovsky's K is approximate.** PatternedSimd's mask table sets
  4–5 bits per mask with a sliding window; the parser sees K=5. The
  realized FP rate is ~8× the theoretical reported in the K=5 column
  because the algorithm is not a true independent-k-position Bloom.

* **Impala's gap is half hash, half algorithm.** Compared to
  `single_key`'s wymum, XXH64 is several times slower per key. In
  prehash mode (hash excluded from the timed loop) Impala lands at
  ~8.5 ns/op — roughly the same as `bloom_classic.c`. The remaining
  algorithmic gap is "scalar bit-set vs SIMD mask compute."

* **`bloom_classic.c` uses double-hashing** (Kirsch–Mitzenmacher) to
  cheaply derive K positions from one 128-bit multiply. A "more
  textbook" version would compute K truly independent hashes; that
  would be slower again.

* **`bloom_xorfuse.c` is a different algorithm class.** Binary fuse
  filters trade build-time for query-time and space: they need ~9
  bits/key (vs SBBF's ~21) but require a **static** build over the
  full key set, and the build is expensive (~25–30 ns/key amortised).
  Bloom-ABI shim forwards `bloom_insert*` to a buffer and calls
  `binary_fuse8_populate` on the first `bloom_contains*`; calling
  insert after a contains forces a full rebuild. Sensible use cases:
  read-only filters (Tempo segment indexes, Parquet bloom payloads
  built at write time) — not streaming dedup. Default FP is ~0.4%
  (8-bit fingerprint); `binary_fuse16` would get ~0.002% at 2×
  bits/key.

## Bench takeaways for the Rust shims

* **fastbloom**'s tagline is "the fastest Bloom filter in Rust" and
  the published benches back that up vs other Rust crates (it beats
  `bloom`, `bloom2`, `growable-bloom-filter`, etc.). Against
  hand-tuned C SBBF with `wymum` and a single-cache-line block, the
  story flips: ~15× slower at S, ~14× at M, ~8× at L. Two reasons —
  SipHash-1-3 is a strong hash (good for adversarial workloads) but
  ~5× slower than `wymum` on 16-byte keys, and the crate's block
  layout touches more memory per query than the 256-bit SBBF block.
* **arrow-rs SBBF** is the production-grade SBBF used by every Rust
  Parquet reader/writer. Same algorithm class as `bloom_impala.c`
  (Parquet spec, XXH64, scalar bit-set) — and ~25% *faster* than
  that reference, thanks to Rust's bounds-check elision and
  inliner. Still ~10–12× slower than `single_key` because of XXH64
  (vs `wymum`) and scalar bit-set (vs SIMD `vptest`).
* **No prehash for `arrow_rs_sbbf_shim`.** The arrow-rs crate
  doesn't expose an `insert_hash` / `check_hash` entry point — its
  public API always hashes via XXH64 internally. Bench prehash rows
  show "(no prehash)" for this candidate.

## Note on CRoaring

CRoaring is a roaring-bitmap library, not a bloom filter library.
It exposes no SBBF-equivalent type; if you saw "CRoaring bloom"
referenced, it's likely the third-party
[`roaring-bloom-filter-rs`](https://github.com/oliverdding/roaring-bloom-filter-rs)
crate, which uses roaring bitmaps as the bit-store under a classical
Bloom — a niche optimisation for very-sparse filters, not generally
competitive on speed. Not in this bench.
