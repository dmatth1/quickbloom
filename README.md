# bloom

Fast Bloom filter implementations for x86_64. Three designs, one ABI,
broad CPU coverage.

## What's here

| File | When to use |
|---|---|
| `bloom_single_key.c` | In-cache filters (< ~10 MB) or contains-heavy workloads — fastest in-cache. |
| `bloom_unified.c` | **Recommended default.** Adds prefetch lookahead; good across the cache hierarchy. |
| `bloom_batched.c` | Out-of-cache filters (≥ ~10 MB), especially insert-heavy or columnar workloads. Exposes an 8-way batched ABI for amortized per-call work. |

All three implement the same single-key ABI; `bloom_batched.c` adds the
optional `bloom_*_batch8` entry points.

## Quick start

```sh
# Build + bench, all three candidates, full sweep across cache regimes
CC=clang python3 bench_all.py

# Just the endpoints (faster — skip M and L)
CC=clang python3 bench_all.py --sizes S,XL

# Just the correctness tests
python3 test_bloom.py
```

`bench_all.py` reports min/median/p90 ns/op at each size for both the
hash-included path (what real users pay) and the pre-hashed path
(algorithm only, for apples-to-apples comparison with published numbers).

## Algorithm

All three implementations are Split Block Bloom Filter
([Apache Parquet spec](https://github.com/apache/parquet-format/blob/master/BloomFilter.md))
variants:

- **`bloom_single_key.c`**: 256-bit blocks, K=8 SBBF salts, wymum hash
  (single 128-bit multiply), 4-way unrolled `bulk_contains`, sequential
  load-or-store per key on the insert path (so hardware store-load
  forwarding handles aliased blocks).
- **`bloom_unified.c`**: same as `single_key.c` plus software prefetch
  lookahead of 8 keys in the bulk paths.
- **`bloom_batched.c`**: 64-bit blocks, K=4 SBBF salts, AVX2 SIMD mask
  construction batched across 8 keys per call. Insert and contains both
  use scalar 8-byte memory ops (no `vpgather` — direct loads are faster
  on every CPU we've measured). Prefetch lookahead 16 keys.

The key hash is `wymum` — one 128-bit multiply on the 16-byte fast path
— with a `fasthash64` fallback for variable-length keys.

## ABI

```c
// Core API (all three implementations)
void*  bloom_new(size_t nbits);
void   bloom_free(void*);
void   bloom_insert(void*, const void* key, size_t len);
int    bloom_contains(void*, const void* key, size_t len);
void   bloom_insert_bulk(void*, const uint8_t* keys, size_t klen, size_t n);
size_t bloom_contains_bulk(void*, const uint8_t* keys, size_t klen, size_t n);

// Pre-hashed bulk (all three; lets you hash once and reuse)
void   bloom_insert_prehash(void*, uint64_t hash);
int    bloom_contains_prehash(void*, uint64_t hash);
void   bloom_insert_prehash_bulk(void*, const uint64_t* hashes, size_t n);
size_t bloom_contains_prehash_bulk(void*, const uint64_t* hashes, size_t n);

// 8-way batched (bloom_batched.c only)
void    bloom_insert_batch8(void*, const uint64_t hashes[8]);
uint8_t bloom_contains_batch8(void*, const uint64_t hashes[8]);  // bitmap of 8 results
void    bloom_insert_batch8_bulk(void*, const uint64_t* hashes, size_t n);
size_t  bloom_contains_batch8_bulk(void*, const uint64_t* hashes, size_t n);
```

## Build target & CPU coverage

Compile flags: `-O3 -mavx2 -mbmi2 -mfma -maes -fPIC -shared`.

Runs on any x86_64 CPU with AVX2 + BMI2:

- **Intel** — Haswell (2013) and later. Haswell, Broadwell, Skylake,
  Coffee Lake, Cascade Lake, Ice Lake, Tiger Lake, Alder Lake, Raptor
  Lake, Sapphire Rapids, Meteor Lake, Emerald Rapids, Granite Rapids,
  Arrow Lake.
- **AMD** — Excavator (2015) and later. Then Zen 1 / Zen 2 / Zen 3 /
  Zen 4 / Zen 5.

Single `.so` covers ~10+ years of x86_64 hardware. We deliberately
target AVX2-only (no AVX-512) — on Sapphire Rapids the AVX-512
frequency throttling penalty exceeds any width advantage for this
workload. Avoiding AVX-512 also dodges the Intel consumer-chip AVX-512
removal (Alder/Raptor/Meteor) entirely.

## Performance

Measured on Intel Xeon Sapphire Rapids @ 2.1 GHz with clang -O3, prehash
mode (hash excluded from the timed loop — comparable to published
cyc/op figures).

| Filter size | single_key | unified | batched |
|---|---:|---:|---:|
| S (128 KB, in L2)  | **~1.1 ns** | ~1.3 ns | ~1.7 ns |
| M (2 MB, in L3)    | ~1.8 ns | **~1.7 ns** | ~1.8 ns |
| L (32 MB, near L3) | ~6–7 ns | ~6–7 ns | **~5.5–6 ns** |
| XL (512 MB, DRAM)  | ~18–20 ns | ~17–19 ns | **~16–17 ns** |

Crossover at the L3 boundary: in-cache `single_key.c` wins; out of L3
`batched.c` wins via batched SIMD mask compute + prefetch lookahead,
even when called through the single-key API.

**Honest caveat**: these numbers are from one CPU. Different
microarchitectures have different `vpmullo` latencies (Haswell ~10 cyc,
Skylake/Cascade Lake ~10, Ice Lake / Sapphire Rapids ~5, Zen 3+ ~3-4),
load buffer sizes, and prefetcher behavior. Run `bench_all.py` on your
target hardware for ground truth. The *relative* ordering (in-cache
favors single_key, out-of-cache favors batched/unified) is expected to
hold across modern x86, but the absolute ns/op will shift.

## Comparison to other designs

In-cache (S = 128 KB filter) contains, prehash path, Sapphire Rapids
@ 2.1 GHz — comparable to published `cyc/op` figures:

| Implementation                | Design                                |  ns/op | cyc/op |
|-------------------------------|---------------------------------------|-------:|-------:|
| **single_key** (this repo)    | SBBF 256-bit, K=8, single-key         | **1.1** | **2.4** |
| **unified** (this repo)       | single_key + prefetch lookahead       |   1.3  |   2.7  |
| **batched** (this repo)       | SBBF 64-bit, K=4, 8-way SIMD batched  |   2.0  |   4.2  |
| Krassovsky PatternedSimd †    | 64-bit blocks, 8-way SIMD batched     |  ~1.2  |   2.5  |
| Apache Impala SBBF †          | 256-bit blocks, single-key            |  ~2.4  |  ~5    |
| fastbloom (Rust, sbbf-AVX2) † | 256-bit blocks, single-key            |  ~3–5  |  ~6–10 |

† Published numbers from upstream benchmark suites, measured on
different hardware. The cycle count is the more portable comparison
since absolute ns/op depends on clock and `vpmullo` latency. Sources:
[save-buffer/bloomfilter_benchmarks](https://github.com/save-buffer/bloomfilter_benchmarks),
[Apache Impala BloomFilter](https://github.com/apache/impala),
[tomtomwombat/fastbloom](https://github.com/tomtomwombat/fastbloom).
All implementations listed are AVX2 (Krassovsky's PatternedSimd uses
`_mm256_i64gather_epi64` + shift-pattern mask; his repo has no AVX-512
variant).

The headline finding: **`single_key` matches Krassovsky's *batched*
published number on cycles, without batching the API.** He needs 8-way
SIMD batching across 8 keys per call to hit 2.5 cyc; we hit 2.4 cyc per
single-key call by using wider blocks (256-bit vs 64-bit) plus a
128-bit-multiply hash (wymum). The wider block amortizes one cache
line's worth of latency over more bits tested per key.

### Variants we tried and rejected

Negative results from the search, all measured against the AVX2
single-key baseline of 1.1 ns/op (2.4 cyc/op) on Sapphire Rapids:

| Variant | Outcome | Why |
|---|---|---|
| **AVX-512 single-key** (zmm SBBF masks) | slower than AVX2 single_key | AVX-512 frequency throttling on SPR exceeds the width gain; SBBF mask compute doesn't saturate Zmm |
| **AVX-512 batched** (zmm 8-way) | slower than AVX2 batched | same — Intel SPR client/server throttles down on heavy AVX-512 |
| **AESENC hash** (replacing wymum) | slower than wymum | port-0 contention with `vpmullo` SBBF mask compute (both prefer p0) |
| **`vpgatherqq` contains** | slower than scalar 8-byte loads | gather is ~12–15 cyc latency on SPR; 8 scalar loads beat it on every CPU we measured |
| **Two parallel `mum` hash chains** | no clear improvement | hash already saturates the multiply pipe |
| **Insert-side collision skip** | slightly slower | branch mispredict cost > duplicate-work savings on random keys |
| **4-way unroll w/ batched store on insert** | 16 false negatives | aliased blocks need serialized load-or-store per key for store-load forwarding (fixed by `single_key`'s sequential L+O+S loop) |
| **Shift-based mask compute** (skip xor-fold) | failed FP rate test | low bits of `r.lo` from a 64×64→128 multiply have weak diffusion |
| **PGO** (clang -fprofile-use) | <2% improvement | code is already mostly loop-free SIMD; PGO mostly helps branchy code |

The negatives are useful context for anyone re-treading this space: SPR's
AVX-512 throttle and the AESENC/vpmullo port-0 collision drive different
tradeoffs than the conventional wisdom on Skylake.

## Methodology

The benchmark harness handles:

- **Two timing modes**: hash+bloom (what a real user pays) and prehash
  (algorithm-only, hash excluded — matches published cyc/op figures
  from Apache Impala, save-buffer/bloomfilter_benchmarks, etc.).
- **Four filter sizes** (`S`/`M`/`L`/`XL`) spanning L2 → DRAM.
- **Random 16-byte keys**, deterministic across runs (SHA-256 of an
  index, truncated). Different seeds for inserted vs unseen sets.
- **Per-phase repeats** with warmup. Reports min/median/p90 across
  repeats; `min` is the headline number (most reproducible).
- **CPU/compiler detection** in the bench header so results are
  self-documenting.

The correctness check (`test_bloom.py` and the bench's `diff_test`):
- Insert `n_insert` random keys, query them all back — must have
  zero false negatives.
- Query `n_query` separately-seeded unseen keys, measure the false
  positive rate — must be below 0.005.

## Choosing the right implementation

```c
// Quick decision tree:
if (filter_size_bytes < L3_size / 8) {
    // In-cache regime
    use(bloom_single_key);
} else if (workload_has_known_8way_batches) {
    // Analytical / bulk path with batched-API benefit
    use(bloom_batched);
} else {
    // Out of L3 but single-key API caller
    use(bloom_unified);  // also a fine default when size isn't known
}
```

If you don't know what to pick, **use `bloom_unified.c`**. It's within
~15% of the best at every size and never embarrassing.

## Files

```
bloom_single_key.c   -- v14 design: SBBF + AVX2 + wymum + 4-way unroll
bloom_unified.c      -- v28: single_key + prefetch lookahead
bloom_batched.c      -- v26: 64-bit blocks + SIMD batched mask + scalar contains
harness.py           -- compile + diff_test + benchmark infrastructure
bench_all.py         -- run benchmarks across cache regimes
test_bloom.py        -- run correctness tests
```

## License

MIT.
