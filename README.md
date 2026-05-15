# quickbloom

Fast Bloom filter implementations for x86_64. Three designs, one ABI.

## What's here

| File | When to use |
|---|---|
| `bloom_single_key.c` | In-cache filters (< ~10 MB) or contains-heavy workloads. Fastest in-cache. |
| `bloom_unified.c` | Good default. Adds prefetch lookahead; competitive across the cache hierarchy. |
| `bloom_batched.c` | Out-of-cache filters (≥ ~10 MB) or columnar workloads. Exposes an 8-way batched ABI. |
| `bloom_classic.c` | Textbook non-blocked Bloom (K=8 double-hashing). Slow baseline; useful for comparison. |
| `bloom_impala.c` | Faithful Apache Parquet/Impala SBBF (scalar, XXH64). External competitor. |
| `bloom_krassovsky.c` | Faithful port of Save Buffer's PatternedSimdBloomFilter. External competitor with batched SIMD + gather. |

`single_key` and `unified` share one source (`bloom_sbbf.c`) and select a
compile-time `PREFETCH_LOOKAHEAD` macro -- two configurations, one
implementation. `bloom_batched.c` adds optional `bloom_*_batch8` entry
points; the others expose the single-key + prehash ABI only.

## Quick start

```sh
# Build + bench, all candidates, full sweep across cache regimes
CC=clang python3 bench_all.py

# Just the endpoints (skip M and L)
CC=clang python3 bench_all.py --sizes S,XL

# Equal-FP comparison: each candidate is sized to hit FP <= 0.001
CC=clang python3 bench_all.py --target-fp 0.001

# Correctness tests (fixed-length + variable-length keys)
python3 test_bloom.py
```

`bench_all.py` reports min/median/p90 ns/op at each size for both the
hash+bloom path (user-visible) and the prehash path (algorithm only,
comparable to published cyc/op).

## Algorithm

All three are Split Block Bloom Filter
([Apache Parquet spec](https://github.com/apache/parquet-format/blob/master/BloomFilter.md))
variants:

- **`bloom_single_key.c`**: 256-bit blocks, K=8 SBBF salts, wymum hash,
  4-way unrolled `bulk_contains`. Insert is a sequential load-or-store
  per key so store-load forwarding handles aliased blocks.
- **`bloom_unified.c`**: same as `single_key.c` plus software prefetch
  lookahead of 8 keys in the bulk paths.
- **`bloom_batched.c`**: 64-bit blocks, K=4 SBBF salts, AVX2 SIMD mask
  compute batched across 8 keys per call. Scalar 8-byte memory ops
  (no `vpgather` — direct loads beat it on every CPU we measured).
  Prefetch lookahead 16.

Hash is `wymum` — one 128-bit multiply on the 16-byte fast path — with
a `fasthash64` fallback for variable-length keys.

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

Runs on any x86_64 with AVX2 + BMI2:

- **Intel** — Haswell (2013) and later.
- **AMD** — Excavator (2015) and later, then Zen 1–5.

One `.so` covers 10+ years of x86_64. AVX2-only by design: on Sapphire
Rapids the AVX-512 frequency throttle exceeds the width win on this
workload, and Intel consumer chips (Alder/Raptor/Meteor) don't have
AVX-512 anyway.

## Performance

Intel Xeon Sapphire Rapids @ 2.1 GHz, clang -O3, prehash mode (hash
excluded from the timed loop):

| Filter size | single_key | unified | batched |
|---|---:|---:|---:|
| S (128 KB, in L2)  | **~1.1 ns** | ~1.3 ns | ~1.7 ns |
| M (2 MB, in L3)    | ~1.8 ns | **~1.7 ns** | ~1.8 ns |
| L (32 MB, near L3) | ~6–7 ns | ~6–7 ns | **~5.5–6 ns** |
| XL (512 MB, DRAM)  | ~18–20 ns | ~17–19 ns | **~16–17 ns** |

Crossover at L3: in-cache `single_key` wins; out of L3 `batched` wins
via SIMD mask compute + prefetch, even through the single-key API.

These numbers are one CPU. `vpmullo` latency varies (Haswell ~10 cyc,
Ice Lake / SPR ~5, Zen 3+ ~3–4), as does prefetcher behavior. Run
`bench_all.py` on your hardware for ground truth — the relative
ordering holds across modern x86, but absolute ns/op shifts.

## Comparison to other designs

`bench_all.py` builds every row in this table from source and measures
them on the host. Hash+bloom path, S = 128 KB filter, contains-miss, all
on Sapphire Rapids @ 2.1 GHz with clang -O3:

| Implementation                | K | Design                                | ns/op | cyc/op | FP rate |
|-------------------------------|---|---------------------------------------|------:|-------:|--------:|
| **single_key** (this repo)    | 8 | SBBF 256-bit, AVX2 mask, wymum hash   | **1.62** | **3.4** | 0.00034 |
| **batched** (this repo)       | 4 | SBBF 64-bit, 8-way SIMD, no gather    |  2.84 |   6.0  | 0.00257 |
| **unified** (this repo)       | 8 | single_key + prefetch                 |  2.03 |   4.3  | 0.00034 |
| `bloom_krassovsky.c`          | 5 | 64-bit blocks, 8-way SIMD + gather    |  1.87 |   3.9  | 0.00346 |
| `bloom_impala.c`              | 8 | 256-bit blocks, scalar bit-set, XXH64 | 17.53 |  36.8  | 0.00034 |
| `bloom_classic.c`             | 8 | textbook double-hashing, no blocking  | 10.33 |  21.7  | 0.00016 |
| fastbloom (Rust, sbbf-AVX2) † | 8 | 256-bit blocks, single-key            |  ~3–5 |  ~6–10 | -       |

`bloom_krassovsky.c` is a faithful port of Save Buffer's
`PatternedSimdBloomFilter` ([source](https://github.com/save-buffer/bloomfilter_benchmarks),
MIT). `bloom_impala.c` is a faithful Apache Parquet SBBF (the same algorithm
Impala and Arrow ship) — scalar bit-set with XXH64, no SIMD. The gap to
`bloom_single_key.c` shows what the SIMD mask compute + wymum hash buys
over a by-the-book SBBF.

**FP rates differ across rows.** `batched` (K=4) and `krassovsky` (K=5,
mask-table) report ~10× higher FP than `single_key` (K=8) at the same
bits/key — so headline ns/op is not directly comparable. To compare at
equal FP, run `bench_all.py --target-fp X`.

† fastbloom is a Rust crate (`tomtomwombat/fastbloom`); not ported here.
The cited number is from its public benchmarks on different hardware. Sources:
[save-buffer/bloomfilter_benchmarks](https://github.com/save-buffer/bloomfilter_benchmarks),
[Apache Impala](https://github.com/apache/impala),
[fastbloom](https://github.com/tomtomwombat/fastbloom). All AVX2;
Krassovsky's PatternedSimd uses `_mm256_i64gather_epi64`, no AVX-512
variant in his repo.

`single_key` matches Krassovsky's batched on cycles without batching
the API. He needs 8-way SIMD across 8 keys per call to hit 2.5 cyc; we
hit 2.4 cyc per single-key call with wider blocks (256-bit vs 64-bit)
and a 128-bit-multiply hash. The wider block amortizes a cache line's
latency over more bits tested per key.

### Variants we tried and rejected

| Variant | Outcome | Why |
|---|---|---|
| AVX-512 single-key (zmm SBBF masks) | slower than AVX2 | freq throttle on SPR exceeds the width gain; SBBF mask compute doesn't saturate Zmm |
| AVX-512 batched (zmm 8-way) | slower than AVX2 | same throttle |
| AESENC hash (replacing wymum) | slower than wymum | port-0 contention with `vpmullo` SBBF mask compute |
| `vpgatherqq` contains | slower than scalar 8-byte loads | ~12–15 cyc gather latency on SPR loses to 8 direct loads |
| Two parallel `mum` hash chains | no improvement | hash already saturates the multiply pipe |
| Insert-side collision skip | slightly slower | branch mispredict cost > dup-work savings on random keys |
| 4-way unroll w/ batched store on insert | 16 false negatives | aliased blocks need serialized L+O+S per key for store-load forwarding |
| Shift-based mask compute (skip xor-fold) | failed FP test | low bits of `r.lo` from a 64×64→128 multiply have weak diffusion |
| PGO (clang -fprofile-use) | <2% | code is mostly loop-free SIMD; PGO mostly helps branchy code |

## Methodology

The harness:

- **Two timing modes**: hash+bloom (user-visible) and prehash (algorithm
  only, comparable to published cyc/op).
- **Four sizes** (`S/M/L/XL`) spanning L2 → DRAM.
- **Random 16-byte keys**, deterministic across runs (SHA-256 of an
  index, truncated). Separate seeds for inserted vs unseen sets.
- **Repeats with warmup**; reports min/median/p90. `min` is the
  headline (most reproducible).
- **CPU/compiler detected** and printed in the bench header.

Correctness (`test_bloom.py` and `diff_test` in the bench):

- Insert `n_insert` keys, query them all back — zero false negatives.
- Query `n_query` separately-seeded unseen keys — FP rate below 0.005.

## Choosing the right implementation

```c
if (filter_size_bytes < L3_size / 8) {
    use(bloom_single_key);          // in-cache
} else if (caller_can_batch_8) {
    use(bloom_batched);              // 8-way batched ABI win
} else {
    use(bloom_unified);              // safe default; also fine when size is unknown
}
```

If unsure, use `bloom_unified.c` — within ~15% of the best at every size.

## Files

```
bloom_sbbf.c         SBBF 256-bit + wymum + 4-way unroll; PREFETCH_LOOKAHEAD macro
bloom_single_key.c   stub: PREFETCH_LOOKAHEAD=0, includes bloom_sbbf.c
bloom_unified.c      stub: PREFETCH_LOOKAHEAD=8, includes bloom_sbbf.c
bloom_batched.c      64-bit blocks + SIMD batched mask, scalar contains
bloom_classic.c      textbook K-hash bloom, no blocking (baseline)
bloom_impala.c       faithful Apache Parquet/Impala SBBF (external competitor)
bloom_krassovsky.c   faithful Save Buffer PatternedSimd (external competitor)
harness.py           compile + diff_test + benchmark infrastructure
bench_all.py         run benchmarks across cache regimes; --target-fp for equal-FP
test_bloom.py        correctness tests (fixed-length and variable-length keys)
```

## License

MIT.
