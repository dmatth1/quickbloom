# Parquet Bloom across the ecosystem: the v14 multiplier opportunity

Companion to `APPLICATIONS.md`. While that document ranks per-system
deployment targets (Scylla, ClickHouse, Redpanda, Reth, Loki, etc.),
this one covers the **layer** beneath many of them: the Parquet bloom
filter implementations shipped by every native columnar engine.

Headline: **the entire native Parquet ecosystem ships scalar SBBF
probes** despite the format being designed for AVX2 (256-bit blocks =
one `__m256i` register). The on-disk format is locked by the Parquet
spec, but the probe *implementation* is not — and every major reader
shipped the slow scalar path because the spec gave a scalar reference
and nobody substituted.

## The three confirmed scalar implementations

All three were pulled directly from upstream on 2026-05-15 and verified
line-for-line equivalent.

### 1. Apache Arrow C++ — the reference

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

**Downstream**: DuckDB (parquet extension links arrow-cpp), ClickHouse
(parquet reader links arrow-cpp), pyarrow → pandas/Polars(pyarrow path),
Apache Drill, Apache Doris.

### 2. arrow-rs (Rust)

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

Same shape. The 8-element mask is precomputed as a `[u32; 8]`, but the
probe is still a sequential 8-iteration loop with early exit. Rust/LLVM
struggles to auto-vectorize this because the early `return false`
disrupts the equivalence with a single horizontal AND.

**Downstream**: Polars (parquet2 path), DataFusion, every Rust analytics
tool reading Parquet.

### 3. Velox (Meta) — Trino/Presto/Spark native execution

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

Line-for-line identical to Apache Arrow C++ aside from the
`__isset.XXHASH` check and `ARROW_PREDICT_FALSE` macro.

**Downstream**: Trino / Presto via Prestissimo, Spark via Velox-native
runtimes, Meta's internal compute. The blast radius for one upstream
Velox commit covers the JVM analytics ecosystem indirectly through
native execution.

## What the spec actually mandates

Per the [Apache Parquet BloomFilter spec][parquet-spec]:

| Element | Fixed by spec |
|---|---|
| Block size | **256 bits** (8 × `uint32_t`) |
| Bits-set-per-block | **8** (one per word) |
| SALT constants | Eight specific 32-bit values (verbatim across all implementations above) |
| Hash function | **XXH64** with seed 0 |
| Bit position formula | `(key * SALT[i]) >> 27` |
| Bucket index | `((hash >> 32) * num_buckets) >> 32` (Lemire fastrange) |

The spec exists to guarantee that any reader gets bit-identical results
across engines — a filter written by Spark must be readable by DuckDB
producing the same answers. **None of these constraints touch the probe
implementation.** Whether you iterate 8 times scalar or do one AVX2
testc, the answer is identical because the underlying bits and mask
math are identical.

## The v14 drop-in

The v14-shape probe for an *identical* Parquet-spec block:

```cpp
bool find_parquet_avx2(uint64_t hash) const {
    uint32_t bucket_index = uint32_t(((hash >> 32) * num_blocks_) >> 32);
    uint32_t key = uint32_t(hash);

    // Mask: vpmulld + vpsrld + vpsllvd produces 8 lanes of (1 << ((key * SALT[i]) >> 27))
    __m256i hash_v = _mm256_set1_epi32(int32_t(key));
    __m256i salt   = _mm256_load_si256(reinterpret_cast<const __m256i*>(SALT));
    __m256i prod   = _mm256_mullo_epi32(hash_v, salt);
    __m256i shift  = _mm256_srli_epi32(prod, 27);
    __m256i ones   = _mm256_set1_epi32(1);
    __m256i mask   = _mm256_sllv_epi32(ones, shift);

    // Probe: one load, one testc -- "are all mask bits set in the block?"
    const __m256i* blk = reinterpret_cast<const __m256i*>(&blocks_[bucket_index]);
    return _mm256_testc_si256(_mm256_load_si256(blk), mask);
}
```

**Bit-identical to the scalar reference.** The mask compute uses the
same SALT constants in the same order; the result is 8 lanes each
containing exactly `1 << ((key * SALT[i]) >> 27)`. The probe uses
`testc` which returns 1 iff `(~block & mask) == 0` — equivalent to
"every mask bit is set in the block."

A diff test against the scalar implementation should produce zero
mismatches across any number of insertions and probes.

## Why nobody did this already

Two reasons, both fixable:

1. **The Parquet spec ships a scalar pseudocode reference**, and the
   first C++/Rust implementations were faithful ports. Once a working
   reference existed, the natural reuse path was to fork it — and the
   fork inherits the scalar probe. Apache Arrow C++ is the original;
   Velox copied it; the pattern propagated.
2. **Compilers don't auto-vectorize** the scalar early-exit loop into
   `_mm256_testc_si256` because the early `return false` semantics
   prevent the compiler from proving equivalence with the all-mask-set
   horizontal check. Even with `-O3 -mavx2`, the generated code is
   8 scalar loads + 8 branches.

Both reasons are removable with a manually-vectorized probe path
behind a runtime CPU feature check. The exact pattern v14 already
implements.

## Blast radius

Three upstream patches cover essentially the entire native Parquet
ecosystem:

| Patch target | Reaches |
|---|---|
| `apache/arrow` cpp/src/parquet/bloom_filter.cc | DuckDB, ClickHouse, pyarrow, pandas, StarRocks, Doris, Apache Drill |
| `apache/arrow-rs` parquet/src/bloom_filter/mod.rs | Polars, DataFusion, every Rust analytics tool |
| `facebookincubator/velox` velox/dwio/parquet/common/BloomFilter.cpp | Trino/Presto via Prestissimo, Spark-native runtimes |

Compare to the per-system patches required for the other v14 targets
(Scylla, RedisBloom, ClickHouse-native-bloom, Reth, Bifrost, btllib).
Parquet bloom is a *layer* underneath multiple table formats (Iceberg,
Delta, Hudi) and multiple engines — one PR multiplies into many
deployments.

## What v14 mitigates in the Parquet displacement debate

Recap from `APPLICATIONS.md`'s section on why systems move away from
Bloom — applied to Parquet bloom specifically:

| Concern driving Parquet bloom displacement | v14 mitigates? |
|---|---|
| Probe CPU on large queries (many row groups × many predicates) | **Yes — 3–5× per probe** |
| Build CPU during writes / compactions | **Yes — 5–10× with `bulk_add`** |
| `WHERE col IN (val1...valN)` cost | **Yes — 4-way `bulk_is_present` amortises N probes** |
| Storage overhead (~10 bits/value per file) | No — same on-disk size |
| FP rate vs alternatives (min/max + page indexes) | No — same algorithm, same FP curve |
| Maintenance cost on UPDATE/MERGE rebuild | Partial — faster `bulk_add` helps |
| Databricks-style displacement (Photon predictive I/O) | No — different layer entirely |

The case for *displacing* Parquet bloom (Photon, liquid clustering) was
driven by storage and maintenance, not probe CPU. v14 doesn't argue
against displacement on those grounds. What v14 does is **widen the
envelope of queries where keeping the bloom beats alternatives** —
because the probe CPU per row group drops 3–5×, the "is bloom worth it
for this workload?" threshold moves up by 3–5× in row-group-count
terms.

For ecosystems that haven't shipped Photon-style alternatives
(Iceberg + Hudi, the entire OSS lakehouse, native engines), this is a
direct improvement to their core scan path with no behavior change.

## The benchmark — see `parquet_port/`

`parquet_port/` contains a standalone microbench using the *real*
Parquet SBBF format (XXH64 hash, spec-locked SALT constants, Lemire
fastrange bucket index) with three probe paths side-by-side:

1. **Scalar Parquet** — line-for-line port of Apache Arrow C++'s
   `FindHash`. The baseline shipped by Arrow / arrow-rs / Velox.
2. **AVX2 Parquet-spec** (v14 drop-in) — same on-disk format, same
   hash, AVX2 testc-based probe. Bit-identical results.
3. **v14 free-standing** — pow2 mask + wymum hash, no spec
   compatibility. Measures the ceiling if the spec weren't locked.

Results sit in `parquet_port/RESULTS.md`. The headline numbers
(Intel Xeon @ 2.8 GHz, 33 MiB L3, virtualised):

| Regime | Footprint | scalar (Arrow/Velox) | AVX2 single-key (v14) | AVX2 4-way (v11) |
|---|---:|---:|---:|---:|
| Small (in-cache) | 0.5 MiB | 12.73 ns | **3.70 ns (3.4×)** | **2.36 ns (5.4×)** |
| Medium (out-of-L3) | 128 MiB | 35.84 ns | 29.18 ns (1.2×) | **22.79 ns (1.6×)** |
| Large (deep DRAM) | 1 GiB | 41.41 ns | 35.90 ns (1.15×) | **27.75 ns (1.5×)** |

**Diff-test: 0 mismatches across 167M (query, filter) pairs** — the
AVX2 path produces bit-identical results to the scalar Arrow C++ /
arrow-rs / Velox reference. The swap is truly drop-in.

The story is different from Scylla in an important way:

- **Scylla** used classical Bloom — K probes to K *different* cache
  lines. The v14 win there was both ALU + cache locality.
- **Parquet** is *already* SBBF — one cache line per probe. The v14
  win for Parquet is *only* the ALU work (scalar 8-iteration loop →
  one `vptestc`). In-cache that's 3-5×; out-of-L3 both paths wait
  for the same DRAM load and the gap narrows to 1.15-1.23× per
  single-key probe.

The **4-way bulk probe** (v11 pattern: same hash, four row-group
filters in parallel) is what actually moves the out-of-L3 number
because the OoO core gets 4 cache misses in flight simultaneously.
That's the API change worth lobbying for upstream — not just
replacing `findHash` with an AVX2 version but exposing a bulk-probe
entry point engines can call when iterating row-group filters.

For a large-table scan workload (`SELECT * WHERE col = 'X'` over
billions of row groups in a 1 GiB-filter-footprint table), the
per-query savings is **~224 µs** vs scalar. At cloud-lakehouse query
volumes that's real CPU time.

[arrow-cc]: https://github.com/apache/arrow/blob/main/cpp/src/parquet/bloom_filter.cc#L348
[arrowrs]: https://github.com/apache/arrow-rs/blob/main/parquet/src/bloom_filter/mod.rs#L241
[velox]: https://github.com/facebookincubator/velox/blob/main/velox/dwio/parquet/common/BloomFilter.cpp#L196
[parquet-spec]: https://github.com/apache/parquet-format/blob/master/BloomFilter.md
