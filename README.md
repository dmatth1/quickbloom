# quickbloom

Fast Bloom filter implementations for x86_64. Three designs, one ABI.

## What's here

| File | When to use |
|---|---|
| `bloom_single_key.c` | In-cache filters (< ~10 MB) or contains-heavy workloads. Fastest in-cache. |
| `bloom_unified.c` | Good default. Adds prefetch lookahead; competitive across the cache hierarchy. |
| `bloom_batched.c` | Out-of-cache filters (≥ ~10 MB) or columnar workloads. Exposes an 8-way batched ABI. |

`single_key` and `unified` share one source (`bloom_sbbf.c`) and select a
compile-time `PREFETCH_LOOKAHEAD` macro — two configurations, one
implementation. `bloom_batched.c` adds optional `bloom_*_batch8` entry
points; the other two expose the single-key + prehash ABI only.

Reference implementations of other designs (Apache Parquet/Impala SBBF,
Save Buffer PatternedSimd, textbook K-hash Bloom) live in
[`comparisons/`](comparisons/README.md) and are built by the same
harness when you pass `--comparisons`.

## Quick start

```sh
# Build + bench the three top-level designs, full sweep across cache regimes
CC=clang python3 bench_all.py

# Include the reference implementations in comparisons/ in the same run
CC=clang python3 bench_all.py --comparisons

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

Intel Xeon @ 2.8 GHz (Cascade Lake-class, 33 MB L3), clang -O3,
hash+bloom contains-miss, ns/op min. See `comparisons/README.md` for
the head-to-head against Impala, Krassovsky, and a textbook baseline
measured the same way.

| Filter size              | single_key | unified | batched |
|--------------------------|-----------:|--------:|--------:|
| S  (128 KB, in L2)       |   **1.95** |    2.61 |    2.75 |
| M  (2 MB,   in L2/L3)    |       5.97 | **4.90**|    6.18 |
| L  (32 MB,  around L3)   |      16.27 |**15.21**|   21.55 |
| XL (512 MB, in DRAM)     |      29.47 |**25.34**|   35.74 |

In-cache `single_key` wins. As the working set leaves L2, prefetch
lookahead in `unified` pulls ahead and stays ahead through DRAM. On
this host with 33 MB L3, `batched` is never the winner — on hosts with
a smaller L3 (or workloads that batch 8 keys per call), the 8-way
batched ABI in `bloom_batched.c` can recover the gap; run the bench
locally to confirm. Prehash-mode numbers (algorithm only, comparable
to published cyc/op) are ~30% lower across the board.

Run `bench_all.py` on your hardware for ground truth — `vpmullo`
latency varies (Haswell ~10 cyc, Ice Lake / SPR ~5, Zen 3+ ~3–4), as
does L3 size and prefetcher behavior.

## Variants we tried and rejected

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
- Variable-length keys (1–64 bytes, deterministic) exercise the
  `fasthash64_var` path in every candidate.

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
harness.py           compile + diff_test + benchmark infrastructure
bench_all.py         run benchmarks across cache regimes; --target-fp / --comparisons
test_bloom.py        correctness tests (fixed-length and variable-length keys)
comparisons/         reference implementations from other designs (impala,
                     krassovsky, classic) plus their own README
```

## License

MIT.
