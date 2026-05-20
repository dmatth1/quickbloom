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

## Measured head-to-head

Run `CC=clang python3 ../bench_all.py --sizes S,M,L --comparisons` to
reproduce the table below on your host. Cross-host variance is real;
the labels in the README's perf table are illustrative.

Sample capture, Intel Xeon @ 2.1 GHz (Sapphire Rapids-class, 260 MB
L3), clang -O3, contains-miss, ns/op min:

| Implementation                | K | S (128 KB) | M (2 MB) | L (32 MB) | FP rate |
|-------------------------------|---|-----------:|---------:|----------:|--------:|
| `single_key` (top-level)      | 8 |   **1.52** | **2.21** |  **6.79** | 0.00030 |
| `unified` (top-level)         | 8 |       1.96 |     2.64 |      7.44 | 0.00030 |
| `batched` (top-level)         | 4 |       2.81 |     3.21 |      7.15 | 0.00239 |
| `bloom_krassovsky.c`          | 5 |       1.83 |     2.74 |      6.94 | 0.00340 |
| `bloom_xorfuse.c`             | 3 |       4.19 |     4.77 |     19.54 | 0.00422 |
| `bloom_classic.c`             | 8 |      10.20 |    13.60 |     19.53 | 0.00013 |
| `bloom_impala.c`              | 8 |      16.61 |    20.20 |     26.03 | 0.00030 |

Build cost (insert\_bulk, ns/key amortised over the populate phase)
shows the static-vs-dynamic split — xorfuse's lazy-build dominates
insert\_bulk time:

| Implementation                | S insert | M insert | L insert | notes |
|-------------------------------|---------:|---------:|---------:|---|
| `single_key`                  |     1.58 |     2.32 |     7.13 | dynamic bloom; insert ≈ contains |
| `bloom_xorfuse.c`             |    24.47 |    31.60 |   209.93 | static xor peel; cache-thrashes on the scratch buffer at L |

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

## Strong competitors not yet wired in

These are well-known alternatives the bench should grow to include,
but each requires non-trivial new build infrastructure (Rust
toolchain, FFI shims, etc.) that's bigger than dropping a `.c` file
in. Listed here so they're tracked, not forgotten:

* **fastbloom** ([tomtomwombat/fastbloom](https://github.com/tomtomwombat/fastbloom),
  Rust). Cites ~3–5 ns/op on its own hardware. Default hasher is
  SipHash-1-3 with full concurrency support. Wiring in requires a
  `cargo build --release` of a tiny crate that exposes the
  `bloom_*` C ABI via `#[no_mangle] extern "C"`, plus harness
  changes to invoke cargo.
* **arrow-rs SBBF** (`parquet::bloom_filter::Sbbf` in
  [apache/arrow-rs](https://github.com/apache/arrow-rs), Rust). The
  production-grade SBBF used by every Rust Parquet reader/writer.
  Same wrapping work as fastbloom; differs in hash (XXH64) and in
  some allocation/serialization details.
* **CRoaring's bloom?** As of the latest [CRoaring](https://github.com/RoaringBitmap/CRoaring)
  source, the library exposes roaring bitmaps but **no bloom filter
  type** — roaring's compressed bitmap is an exact-membership data
  structure, not probabilistic. If you've seen "CRoaring bloom"
  cited, it's likely the third-party
  [`roaring-bloom-filter-rs`](https://github.com/oliverdding/roaring-bloom-filter-rs)
  crate which uses roaring bitmaps as the bit-store under a classical
  Bloom — a niche optimisation for very-sparse filters, not generally
  competitive on speed. Worth confirming with the original source
  before adding.
