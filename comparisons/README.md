# Reference comparisons

External Bloom filter implementations built and benchmarked by the
same harness as quickbloom. Numbers come from running them on the
host you ran on — not asserted figures from an upstream README.

```sh
CC=clang python3 ../bench_all.py --comparisons   # bench
python3 ../test_bloom.py                         # correctness
```

## What's here

| File | Source | Algorithm |
|---|---|---|
| `bloom_impala.c` | parquet-format/BloomFilter.md | SBBF, 256-bit blocks, K=8, Parquet salts, **scalar bit-set**, **XXH64** hash. |
| `bloom_krassovsky.c` | [save-buffer/bloomfilter_benchmarks](https://github.com/save-buffer/bloomfilter_benchmarks) (MIT) | PatternedSimd: 64-bit blocks, 1024-entry mask table (4–5 bits set, sliding window), AVX2 8-way batched, `vpgatherqq` contains. **Batched-only** — no comparably-tuned single-key path. |
| `bloom_classic.c` | textbook | K=8 double-hashing (`h_i = h1 + i·h2`), non-blocked. Pre-SBBF baseline. |
| `bloom_xorfuse.c` | [FastFilter/xor_singleheader](https://github.com/FastFilter/xor_singleheader) (Apache 2.0) | Graf+Lemire binary fuse filter (8-bit fingerprints, 3-hash xor). **Static**: must be built from a known key set; this file is a shim exposing the `bloom_*` ABI by buffering inserts and lazy-building on first contains. |
| `fastbloom_shim/` | [tomtomwombat/fastbloom](https://github.com/tomtomwombat/fastbloom) (MIT/Apache-2.0) | **Rust cdylib**. K=8 multi-block Bloom, SipHash-1-3 hasher. |
| `arrow_rs_sbbf_shim/` | [apache/arrow-rs](https://github.com/apache/arrow-rs) `parquet::bloom_filter::Sbbf` (Apache 2.0) | **Rust cdylib**. Production SBBF used by every Rust Parquet reader/writer. K=8, XXH64, scalar bit-set (same algorithm as `bloom_impala.c`). No prehash API exposed. |

The Rust shims live in subdirectories with their own `Cargo.toml`;
the harness detects them by the presence of `Cargo.toml` and builds
with `cargo build --release`. Without the Rust toolchain those
candidates are skipped; the C-only path keeps working.

## Notes on individual candidates

* **`bloom_krassovsky.c` isn't directly comparable to quickbloom.**
  The design is fundamentally batched — `vpgatherqq` loads 4 blocks
  per instruction, with no equally-tuned single-key path. The numbers
  in the head-to-head are measured through its 8-way
  `bloom_*_batch8_bulk` entry points. For per-key workloads (caches,
  point lookups, streaming dedup) it doesn't apply. K=5 also produces
  ~10× higher FP than K=8 at equal sizing. `vpgatherqq` cost is also
  CPU-dependent: on Cascade Lake quickbloom is multiple times faster;
  on Sapphire Rapids the gap narrows.

* **`bloom_xorfuse.c` is a different algorithm class.** ~9 bits/key
  (vs SBBF's ~21) but 3 cache lines per probe and a static build
  (~25–210 ns/key) on the full key set. Sensible for read-only
  filters (Tempo segment indexes, Parquet bloom payloads built at
  write time), wrong shape for streaming dedup.

* **`bloom_impala.c` and `arrow_rs_sbbf_shim/` share an algorithm.**
  Both are Parquet-spec SBBF + XXH64 + scalar bit-set. The Rust shim
  is the slower of the two in the current bench (arrow_rs hash+bloom
  M ≈ 29 ns; impala prehash M ≈ 11 ns + XXH64 ≈ 3.5 ns ≈ 14.6 ns),
  presumably from the dyn-dispatch path inside `parquet::bloom_filter`
  plus the lack of a public prehash entry point. Both lose to
  quickbloom by ~8× in prehash on the scalar-vs-SIMD bit-test alone.

* **`fastbloom`'s "fastest Bloom in Rust" claim holds Rust-vs-Rust.**
  Against a `wymum`-hashed C SBBF it's ~7–10× slower: SipHash-1-3
  (cryptographic-strength, ~4× slower than `wymum`) plus a non-SBBF
  multi-block layout.

* **No `arrow_rs` prehash.** The parquet crate doesn't expose
  `insert_hash` / `check_hash`. Its hash+bloom numbers stand alone;
  see the perf table in the top-level README.

## Note on CRoaring

CRoaring is a roaring-bitmap library, not a bloom filter library. It
exposes no SBBF-equivalent type. If you saw "CRoaring bloom"
referenced, it's likely the third-party
[`roaring-bloom-filter-rs`](https://github.com/oliverdding/roaring-bloom-filter-rs)
crate, which uses roaring bitmaps as the bit-store under a classical
Bloom — a niche optimisation for very-sparse filters, not generally
competitive on speed. Not in this bench.
