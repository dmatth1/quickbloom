# quickbloom

Fast Bloom filter implementations for x86_64. Three designs, one ABI.

## What's here

The three quickbloom designs:

| File | Role | When to use |
|---|---|---|
| `bloom_single_key.c` | quickbloom: **single_key** | In-cache filters (< ~10 MB) or contains-heavy workloads. Fastest in-cache. |
| `bloom_unified.c`    | quickbloom: **unified**    | Good default. Adds prefetch lookahead; competitive across the cache hierarchy. |
| `bloom_batched.c`    | quickbloom: **batched**    | Out-of-cache filters (≥ ~10 MB) or columnar workloads. Exposes an 8-way batched ABI. |

`single_key` and `unified` share one source (`bloom_sbbf.c`) and select a
compile-time `PREFETCH_LOOKAHEAD` macro. `bloom_batched.c` adds optional `bloom_*_batch8` entry
points; the other two expose the single-key + prehash ABI only.

Reference implementations of other designs (Apache Parquet/Impala SBBF,
Save Buffer PatternedSimd, textbook K-hash Bloom) live in
[`comparisons/`](comparisons/README.md) and are built by the same
harness when you pass `--comparisons`. The head-to-head table below
includes them.

## Quick start

### Use it in your project

```sh
git clone https://github.com/dmatth1/quickbloom
cd quickbloom
make             # builds build/libquickbloom.{a,so}
sudo make install            # installs to /usr/local by default
                             # override with PREFIX=/your/path
```

```c
#include <quickbloom.h>

void* f = qb_unified_new(qb_estimate_bits(100000, 0.01));   // 100k items, 1% FP
qb_unified_insert(f, "hello", 5);
if (qb_unified_contains(f, "hello", 5)) { /* probably present */ }
qb_unified_free(f);
```

Link with `-lquickbloom -lm` (or use pkg-config: `pkg-config --cflags
--libs quickbloom`). See `examples/hello_quickbloom.c` for a complete
runnable example (`make example && ./build/hello_quickbloom`).

### Run the benchmarks and tests

```sh
make test                                       # native C correctness tests
CC=clang python3 bench_all.py                   # full Python bench sweep
CC=clang python3 bench_all.py --comparisons     # include reference impls
CC=clang python3 bench_all.py --sizes S,XL      # just the endpoints
CC=clang python3 bench_all.py --target-fp 0.001 # equal-FP comparison
python3 test_bloom.py                           # Python correctness tests
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

The three variants live side-by-side in one library with namespaced
symbols, so you can pick a different variant for different code paths in
the same binary. `<NS>` below is one of `qb_single_key`, `qb_unified`,
`qb_batched`. Full declarations are in `quickbloom.h`.

```c
// Core API (all three variants)
void*  <NS>_new(size_t nbits);              // NULL on alloc failure
void   <NS>_free(void* p);                  // accepts NULL
void   <NS>_insert(void* p, const void* key, size_t len);
int    <NS>_contains(void* p, const void* key, size_t len);
void   <NS>_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n);
size_t <NS>_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n);

// Pre-hashed (all three; lets you hash once and reuse)
void   <NS>_insert_prehash(void* p, uint64_t hash);
int    <NS>_contains_prehash(void* p, uint64_t hash);
void   <NS>_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n);
size_t <NS>_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n);

// 8-way batched (qb_batched only)
void    qb_batched_insert_batch8(void* p, const uint64_t hashes[8]);
uint8_t qb_batched_contains_batch8(void* p, const uint64_t hashes[8]);  // bitmap of 8
void    qb_batched_insert_batch8_bulk(void* p, const uint64_t* hashes, size_t n);
size_t  qb_batched_contains_batch8_bulk(void* p, const uint64_t* hashes, size_t n);

// Sizing helper (variant-agnostic)
size_t qb_estimate_bits(size_t n, double fp);   // bits to hold ~n items at fp
```

### Thread safety

- Concurrent **reads** (`<NS>_contains*`, `qb_batched_contains_batch8*`)
  are safe.
- Concurrent **writes** (`<NS>_insert*`) are **not** safe; the bit-set
  is done with a non-atomic load-or-store. Callers that insert
  concurrently must provide their own synchronisation.
- Reads concurrent with writes may see partial state but never a false
  negative once the write completes.

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

> Numbers below are illustrative and **host-dependent**. Crossover
> points between variants shift with L3 size, DRAM latency, prefetcher
> aggressiveness, and load on neighbour cores. Run `make bench` (or
> `python3 bench_all.py`) on your target hardware to see what's true
> for your CPU.

Captured on Intel Xeon @ 2.1 GHz (Sapphire Rapids-class, 260 MB L3),
clang -O3, contains-miss hash+bloom path, ns/op min:

**quickbloom across the cache hierarchy:**

| Filter size              | quickbloom: single_key | quickbloom: unified | quickbloom: batched |
|--------------------------|-----------------------:|--------------------:|--------------------:|
| S  (128 KB, in L2)       |               **1.52** |                1.96 |                2.81 |
| M  (2 MB,   in L3)       |               **2.25** |                2.69 |                3.28 |
| L  (32 MB,  in L3 here)  |               **6.62** |                7.17 |                7.90 |

On a smaller-L3 host (~33 MB), L falls past L3 and `batched` /
`unified` pull ahead at that regime — that's the regime they were
designed for. Prehash-mode numbers (algorithm only) are 20–40% lower
across the board; see `bench_all.py` output.

**Head-to-head vs reference designs** (same metric):

| Implementation             | K | S (128 KB) | M (2 MB) | L (32 MB) | FP rate |
|----------------------------|---|-----------:|---------:|----------:|--------:|
| **quickbloom: single_key** | 8 |   **1.64** | **2.42** |      7.16 | 0.00030 |
| quickbloom: unified        | 8 |       1.98 |     2.70 |      7.58 | 0.00030 |
| quickbloom: batched        | 4 |       2.84 |     3.34 |      7.00 | 0.00239 |
| `krassovsky`               | 5 |       1.84 |     2.93 |   **6.51**| 0.00340 |
| `xorfuse` (binary fuse)    | 3 |       4.24 |     4.76 |     19.52 | 0.00422 |
| `classic`                  | 8 |      10.15 |    13.13 |     19.38 | 0.00013 |
| `impala`                   | 8 |      16.49 |    20.11 |     26.27 | 0.00030 |
| `arrow_rs` SBBF (Rust)     | 8 |      19.66 |    24.15 |     31.87 | 0.00030 |
| `fastbloom` (Rust)         | 8 |      25.07 |    35.12 |     56.69 | 0.00012 |

Reproduce: `CC=clang python3 bench_all.py --sizes S,M,L --comparisons`.

- **`krassovsky`** (Save Buffer's `PatternedSimd`) is the closest
  competitor — within 10–20% on this CPU and slightly *ahead* of
  `single_key` at the L size (where its `vpgatherqq` lookup pays off
  on Sapphire Rapids). FP rate is ~10× higher (K=5); at equal-FP it
  falls behind.
- **`xorfuse`** (Graf+Lemire binary fuse filter) is a different class:
  static set, ~9 bits/key (vs SBBF's ~21), but probes 3 cache lines
  per contains (vs SBBF's 1) so it loses on query latency. Build cost
  is also ~10–30× higher (~25–210 ns/key vs ~2–7 ns/key for SBBF).
  Right answer for read-only filters; wrong answer for streaming.
- **`classic`** (textbook K-hash Bloom) loses ~3–7× because K=8 scattered
  probes pay K cache-line tag-checks per query; SBBF pays 1.
- **`impala`** (Apache Parquet reference SBBF, scalar) loses ~10× in
  hash+bloom and ~7× in prehash. Half the penalty is XXH64 vs `wymum`,
  half is scalar bit-set vs SIMD `vptest`.
- **`arrow_rs`** (apache/arrow-rs `parquet::bloom_filter::Sbbf`,
  Rust). The production SBBF every Rust Parquet reader uses. Same
  XXH64 + scalar SBBF as `impala` but ~25% faster than that
  reference — Rust's bounds-check elision and inliner help. Still
  ~10–12× slower than `single_key`. No prehash API exposed.
- **`fastbloom`** (tomtomwombat/fastbloom, Rust). Markets itself as
  "the fastest Bloom filter in Rust" — and may well be vs other Rust
  options, but loses ~15× to `single_key` here. Two reasons: it uses
  SipHash-1-3 (cryptographic-strength hash, ~5× slower than `wymum`
  on 16-byte keys), and its block layout probes more memory per
  query than a 256-bit-block SBBF. The crate's claim is about
  Rust-vs-Rust, not Rust-vs-hand-tuned-C.

**Head-to-head vs other designs**, at S (128 KB, in L2), contains-miss,
ns/op min (cyc/op at 2.8 GHz):

| Implementation             | K | hash+bloom | prehash | cyc/op | FP rate |
|----------------------------|---|-----------:|--------:|-------:|--------:|
| **quickbloom: single_key** | 8 |   **1.99** |**1.44** |**4.0** | 0.00034 |
| **quickbloom: unified**    | 8 |       2.56 |    1.74 |    4.9 | 0.00034 |
| **quickbloom: batched**    | 4 |       2.72 |    2.09 |    5.9 | 0.00257 |
| `krassovsky`               | 5 |       5.63 |    4.96 |   13.9 | 0.00346 |
| `classic`                  | 8 |       9.40 |    8.49 |   23.8 | 0.00016 |
| `impala`                   | 8 |      17.56 |    8.51 |   23.8 | 0.00034 |

Reproduce: `CC=clang python3 bench_all.py --sizes S --comparisons`.

- **Impala's 9× hash+bloom gap shrinks to 2× in prehash mode** —
  half the penalty is XXH64 vs wymum, half is scalar bit-set vs SIMD
  mask compute.
- **Krassovsky's `vpgatherqq` is CPU-sensitive.** ~3× vs scalar loads
  on Cascade Lake, ~16% on Sapphire Rapids. `single_key` wins on both.
- **FP rates differ.** K=4/5 rows report ~10× higher FP than K=8 at
  equal bits/key. Pass `--target-fp X` to normalize; ordering doesn't
  change on this CPU.

Not ported: fastbloom (Rust crate `tomtomwombat/fastbloom`, published
~3–5 ns/op on its own hardware).

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
    use(qb_single_key_*);            // in-cache
} else if (caller_can_batch_8) {
    use(qb_batched_*);                // 8-way batched ABI win
} else {
    use(qb_unified_*);                // safe default; also fine when size is unknown
}
```

If unsure, use `qb_unified_*` — within ~15% of the best at every size.

## Files

```
quickbloom.h         public header declaring all three variants' APIs
bloom_sbbf.c         shared SBBF implementation (256-bit + wymum + 4-way unroll);
                     instantiated by bloom_single_key.c and bloom_unified.c
                     with different QB_NS / PREFETCH_LOOKAHEAD
bloom_single_key.c   entry point: QB_NS=qb_single_key, PREFETCH_LOOKAHEAD=0
bloom_unified.c      entry point: QB_NS=qb_unified,    PREFETCH_LOOKAHEAD=8
bloom_batched.c      entry point: QB_NS=qb_batched (64-bit blocks + batched SIMD)
qb_util.c            variant-agnostic helpers (qb_estimate_bits)
Makefile             build libquickbloom.{a,so} + test + example; supports install
quickbloom.pc.in     pkg-config template; installed as quickbloom.pc
examples/            minimal usage example
test/                native C correctness tests (mirrors test_bloom.py)
.github/workflows/   CI: build + test on gcc and clang
harness.py           compile + diff_test + benchmark infrastructure
bench_all.py         benchmark sweep across cache regimes; --target-fp / --comparisons
test_bloom.py        Python correctness tests (fixed-length and variable-length keys)
comparisons/         reference implementations from other designs (impala,
                     krassovsky, classic) plus their own README
```

The shared library follows Linux SONAME convention:
`libquickbloom.so.0.1.0` is the real file, `libquickbloom.so.0` is the
SONAME (ABI generation) symlink, `libquickbloom.so` is the linker
name. A `quickbloom.pc` pkg-config file is installed alongside.

## Versioning

Current release: `0.1.0`. Version macros are exposed in `quickbloom.h`
(`QUICKBLOOM_VERSION_{MAJOR,MINOR,PATCH}`, `QUICKBLOOM_VERSION_STRING`).
We follow semantic versioning for the C ABI declared in `quickbloom.h`:
breaking changes bump major, additive changes bump minor, fixes bump
patch.

## License

MIT — see [LICENSE](LICENSE).
