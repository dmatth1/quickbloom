# Porting arrow-go's AVX2 SBBF probe to the C++ family

arrow-go shipped AVX2/SSE4/NEON Parquet SBBF probes in 18.3.0
([PR #336](https://github.com/apache/arrow-go/pull/336)). Apache
Impala and Kudu shipped AVX2 SBBF probes years before that. The
other three big native readers — arrow-cpp, arrow-rs, and Velox —
still ship the scalar pseudocode reference, line-for-line identical
across all three.

The on-disk format is locked by the Parquet spec; the probe
*implementation* is not. An AVX2 drop-in is bit-identical to the
scalar reference and 3–5× faster in-cache on the probe
microbenchmark. This document is the upstream pitch.

## The three still-scalar implementations

Pulled directly from upstream on 2026-05-15.

### Apache Arrow C++ — the reference

[`apache/arrow/cpp/src/parquet/bloom_filter.cc:348`][arrow-cc]:

```cpp
bool BlockSplitBloomFilter::FindHash(uint64_t hash) const {
    const uint32_t bucket_index = static_cast<uint32_t>(
        ((hash >> 32) * (num_bytes_ / kBytesPerFilterBlock)) >> 32);
    const uint32_t key = static_cast<uint32_t>(hash);
    const uint32_t* bitset32 = reinterpret_cast<const uint32_t*>(data_->data());

    for (int i = 0; i < kBitsSetPerBlock; ++i) {
        const uint32_t mask = UINT32_C(0x1) << ((key * SALT[i]) >> 27);
        if (ARROW_PREDICT_FALSE(0 ==
                (bitset32[kBitsSetPerBlock * bucket_index + i] & mask))) {
            return false;
        }
    }
    return true;
}
```

Eight dependent loads, early-exit branch per iteration. No AVX2.

**Downstream**: DuckDB, ClickHouse (parquet path), pyarrow → pandas /
Polars (pyarrow path), Apache Drill, Apache Doris, StarRocks.

### arrow-rs (Rust)

[`apache/arrow-rs/parquet/src/bloom_filter/mod.rs:241`][arrowrs]:

```rust
fn check(&self, hash: u32) -> bool {
    let mask = Self::mask(hash);
    for i in 0..8 {
        if self[i] & mask[i] == 0 {
            return false;
        }
    }
    true
}
```

Same shape. The mask is precomputed as `[u32; 8]`, but the probe is
still sequential with early exit — LLVM can't fold this into a
horizontal AND because the early `return false` blocks the
equivalence proof.

**Downstream**: Polars (parquet2 path), DataFusion, every Rust
analytics tool reading Parquet.

### Velox — Trino / Presto / Spark-native

[`facebookincubator/velox/velox/dwio/parquet/common/BloomFilter.cpp:196`][velox]:

```cpp
bool BlockSplitBloomFilter::findHash(uint64_t hash) const {
    const uint32_t bucketIndex = static_cast<uint32_t>(
        ((hash >> 32) * (numBytes_ / kBytesPerFilterBlock)) >> 32);
    const uint32_t key = static_cast<uint32_t>(hash);
    const uint32_t* bitset32 = reinterpret_cast<const uint32_t*>(data_->as<char>());

    for (int i = 0; i < kBitsSetPerBlock; ++i) {
        const uint32_t mask = UINT32_C(0x1) << ((key * SALT[i]) >> 27);
        if (0 == (bitset32[kBitsSetPerBlock * bucketIndex + i] & mask)) {
            return false;
        }
    }
    return true;
}
```

Line-for-line identical to Apache Arrow C++ aside from `__isset.XXHASH`
and the `ARROW_PREDICT_FALSE` macro.

**Downstream**: Trino / Presto via Prestissimo, Spark via Velox-native
runtimes, Meta's internal compute.

## What the spec mandates

Per the [Apache Parquet BloomFilter spec][parquet-spec]:

| Element              | Fixed by spec                                           |
|----------------------|---------------------------------------------------------|
| Block size           | **256 bits** (8 × `uint32_t`)                           |
| Bits-set-per-block   | **8** (one per word)                                    |
| SALT constants       | Eight specific 32-bit values                            |
| Hash function        | **XXH64** with seed 0                                   |
| Bit position formula | `(key * SALT[i]) >> 27`                                 |
| Bucket index         | `((hash >> 32) * num_buckets) >> 32` (Lemire fastrange) |

The spec exists to guarantee bit-identical results across engines — a
filter written by Spark must read the same in DuckDB. **None of these
constraints touch the probe implementation.** Whether you iterate
8 times scalar or do one AVX2 testc, the answer is identical because
the underlying bits and mask math are.

## The AVX2 drop-in

Same on-disk format. Same SALT. Same XXH64. Same bucket index. Only
the probe changes:

```cpp
bool find_avx2(uint64_t hash) const {
    uint32_t bucket_index = uint32_t(((hash >> 32) * num_blocks_) >> 32);
    uint32_t key = uint32_t(hash);

    // Mask: 8 lanes of (1 << ((key * SALT[i]) >> 27))
    __m256i hash_v = _mm256_set1_epi32(int32_t(key));
    __m256i salt   = _mm256_load_si256(reinterpret_cast<const __m256i*>(SALT));
    __m256i prod   = _mm256_mullo_epi32(hash_v, salt);
    __m256i shift  = _mm256_srli_epi32(prod, 27);
    __m256i ones   = _mm256_set1_epi32(1);
    __m256i mask   = _mm256_sllv_epi32(ones, shift);

    // Probe: one load + testc — "all mask bits set in the block?"
    const __m256i* blk = reinterpret_cast<const __m256i*>(&blocks_[bucket_index]);
    return _mm256_testc_si256(_mm256_load_si256(blk), mask);
}
```

About a dozen instructions, no branches, ~7 cycles in the hot path.

**Bit-identical to the scalar reference.** `_mm256_mullo_epi32(hash_v, salt)`
produces eight 32-bit products each equal to scalar `key * SALT[i]`;
`_mm256_testc_si256` returns 1 iff `(~block & mask) == 0` — exactly
the condition "every mask bit is set in the block."

## Why the C++ family is still scalar

Other implementations did SIMD this years ago — Impala and Kudu in
their native bloom code, arrow-go in 18.3.0. The C++ family stuck
with the scalar reference for two reasons:

1. **The spec ships a scalar pseudocode reference**, and the first
   C++/Rust implementations were faithful ports. Apache Arrow C++
   is the original; Velox copied it; arrow-rs ported it. The fork
   chain inherited the scalar shape.
2. **Compilers don't auto-vectorize the early-exit loop** into
   `_mm256_testc_si256`. The early `return false` prevents the
   compiler from proving equivalence with a horizontal AND-test
   across the whole 256-bit block. Even with `-O3 -mavx2`, the
   generated code is 8 scalar loads + 8 branches.

Both go away with a manually-vectorized probe behind a runtime AVX2
feature check — which is exactly what arrow-go did.

## Blast radius

Three upstream patches cover essentially the entire native Parquet
ecosystem:

| Patch target                                                        | Reaches                                                             |
|---------------------------------------------------------------------|---------------------------------------------------------------------|
| `apache/arrow` cpp/src/parquet/bloom_filter.cc                      | DuckDB, ClickHouse, pyarrow, pandas, StarRocks, Doris, Apache Drill |
| `apache/arrow-rs` parquet/src/bloom_filter/mod.rs                   | Polars, DataFusion, every Rust analytics tool                       |
| `facebookincubator/velox` velox/dwio/parquet/common/BloomFilter.cpp | Trino / Presto via Prestissimo, Spark-native runtimes               |

Parquet bloom is a *layer* underneath multiple table formats
(Iceberg, Delta, Hudi) and multiple engines — one PR multiplies into
many deployments.

## What this fixes — and what it doesn't

The Parquet ecosystem moved through a "should we displace bloom?"
cycle recently — Databricks deprecated Parquet bloom in Delta in
favour of Photon predictive I/O + liquid clustering. That displacement
was driven by storage and maintenance cost, not probe CPU.

| Concern                                          | AVX2 probe mitigates?                              |
|--------------------------------------------------|----------------------------------------------------|
| Probe CPU on large queries                       | **Yes — 3–5× in-cache, 1.15–1.5× out-of-L3**       |
| `WHERE col IN (val1...valN)` cost                | **Yes — 4-way bulk amortises N values**            |
| Storage overhead (~10 bits/value extra per file) | No                                                 |
| FP rate vs alternatives (min/max + page indexes) | No                                                 |
| Maintenance cost on UPDATE/MERGE rebuild         | No                                                 |
| Photon-style displacement                        | No — different layer entirely                      |

The case isn't "displace the displacers." It's that the AVX2 path
**widens the envelope of queries where keeping the bloom beats
alternative skipping**. The "is bloom worth it for this workload?"
threshold moves up in row-group-count terms.

For ecosystems without Photon-style alternatives — Iceberg, Hudi,
the entire OSS lakehouse, native engines like DuckDB and ClickHouse —
this is a direct improvement to the core scan path with zero
behaviour change.

## The benchmark — see `parquet_port/`

`parquet_port/` contains a standalone microbench using the *real*
Parquet SBBF format (XXH64, spec SALT, Lemire fastrange) with three
probe paths side-by-side:

1. **Scalar** — line-for-line port of Apache Arrow C++'s `FindHash`.
   The baseline shipped by Arrow / arrow-rs / Velox.
2. **AVX2 single-key** — same on-disk format, AVX2 testc-based probe.
3. **AVX2 4-way bulk** — one query hash probed against four
   row-group filters in parallel; overlaps cache misses for
   out-of-L3 wins.

Headline numbers from `parquet_port/RESULTS.md` (Intel Xeon @ 2.8 GHz,
33 MiB L3, virtualised):

| Regime             | Footprint | scalar (Arrow/Velox) |   AVX2 single-key   |   AVX2 4-way bulk   |
|--------------------|----------:|---------------------:|--------------------:|--------------------:|
| Small (in-cache)   |   0.5 MiB |             12.73 ns | **3.70 ns (3.4×)**  | **2.36 ns (5.4×)**  |
| Medium (out-of-L3) |   128 MiB |             35.84 ns |     29.18 ns (1.2×) | **22.79 ns (1.6×)** |
| Large (deep DRAM)  |     1 GiB |             41.41 ns |    35.90 ns (1.15×) | **27.75 ns (1.5×)** |

**Diff-test: 0 mismatches across 167M (query, filter) pairs** — the
AVX2 path produces bit-identical results to the scalar reference. The
swap is truly drop-in.

In-cache the gain is pure ALU work — 8 dependent loads + branches
becomes two SIMD ops after one cache-line load. Out-of-L3 both paths
wait for the same DRAM load and the per-probe gap narrows; the 4-way
bulk probe widens it back by overlapping four cache misses in flight.

For a large-table scan (`SELECT * WHERE col = 'X'` over thousands of
row groups in a 1 GiB-filter-footprint table), the per-query saving
is **~224 µs** vs scalar.

[arrow-cc]: https://github.com/apache/arrow/blob/main/cpp/src/parquet/bloom_filter.cc#L348
[arrowrs]: https://github.com/apache/arrow-rs/blob/main/parquet/src/bloom_filter/mod.rs#L241
[velox]: https://github.com/facebookincubator/velox/blob/main/velox/dwio/parquet/common/BloomFilter.cpp#L196
[parquet-spec]: https://github.com/apache/parquet-format/blob/master/BloomFilter.md
