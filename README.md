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

**The headline**: with hashing equalised (prehash mode below),
`quickbloom: single_key` is the fastest single-key SBBF probe kernel
on AVX2 x86_64 across all three cache regimes against every
competitor in this bench. Only `PatternedSimd` (`krassovsky`) is in
the same order of magnitude — and only in its batched-only API
shape, which doesn't apply to per-key callers.

Captured on Intel Xeon @ 2.8 GHz (Cascade Lake-class, 33 MB L3),
clang -O3, contains-miss, min ns/op.

### quickbloom — full `Test([]byte)` path

What a user actually pays end-to-end (`wymum` hash + SBBF probe):

| Variant         | K | API shape           | S (128 KB) | M (2 MB) | L (32 MB) |
|-----------------|---|---------------------|-----------:|---------:|----------:|
| **single_key**  | 8 | single-key          |   **1.75** |     5.17 | **13.29** |
| unified         | 8 | single-key          |       2.46 | **4.61** |     16.65 |
| batched         | 4 | single-key + batch8 |       2.65 |     5.44 |     21.19 |

### Head-to-head, algorithm-only (prehash)

Apples-to-apples: hashing factored out on both sides, so only probe
geometry and cache behavior remain. Add the per-key hash cost from
the table further down to reconstruct hash+bloom latency for any
candidate.

| Implementation             | K | API shape           | S (128 KB) | M (2 MB) | L (32 MB) | FP rate |
|----------------------------|---|---------------------|-----------:|---------:|----------:|--------:|
| **quickbloom: single_key** | 8 | single-key          |   **1.27** | **2.87** | **10.98** | 0.00030 |
| quickbloom: unified        | 8 | single-key          |       1.48 |     3.07 |     14.29 | 0.00030 |
| quickbloom: batched        | 4 | single-key + batch8 |       1.90 |     3.62 |     14.72 | 0.00239 |
| `xorfuse` (binary fuse)    | 3 | **static set**      |       3.97 |     4.91 |     25.83 | 0.00422 |
| `krassovsky`               | 5 | **batched-only**    |       4.83 |     5.97 |     20.52 | 0.00340 |
| `impala`                   | 8 | single-key          |       7.95 |    10.24 |     26.46 | 0.00030 |
| `classic`                  | 8 | single-key          |       7.77 |    11.21 |     34.07 | 0.00013 |
| `fastbloom` (Rust)         | 8 | single-key          |      10.21 |    15.28 |     62.41 | 0.00010 |
| `arrow_rs` SBBF (Rust) †   | 8 | single-key          |       n/a‡ |     n/a‡ |      n/a‡ | 0.00030 |

† `arrow_rs` descends from the `impala` reference (same Parquet-spec
SBBF, same XXH64). Absent a prehash API, you'd expect the kernel to
land near `impala`'s 7.95 / 10.24 / 26.46 ns/op — the gap to
`single_key` is API surface and hash choice, not kernel quality.

‡ The parquet crate doesn't expose `insert_hash` / `check_hash`;
only the bytes-in path with internal XXH64. Its full hash+bloom
numbers are 18.22 / 23.83 / 69.20 ns at S/M/L.

### Per-key hash cost

Kernel cost of each hash function on 16-byte keys, same compiler
flags, same host. Reproduce with `make bench-hash` (source:
[`tools/bench_hash.c`](tools/bench_hash.c)):

| Hash         | Used by                       | ns/op (min) |
|--------------|-------------------------------|------------:|
| `wymum16`    | `quickbloom: *`               |    **0.90** |
| `XXH64`      | `impala`, `arrow_rs`          |        2.80 |
| `SipHash-1-3`| `fastbloom`                   |        3.40 |

Kernel-only — real-world cost in `fastbloom` is higher than the
3.40 ns above because the Rust `Hasher` trait adds per-call
dispatch and the seed is loaded from filter state on every call.
The bench's full-`Test` row for `fastbloom` shows ~15 ns of
end-to-end hash overhead on this host, vs `single_key`'s ~0.5 ns.

### Equal-FP cost (`--target-fp 0.001`)

What the comparison looks like when each candidate is sized to hit
FP ≤ 0.001 (`quickbloom: single_key`'s natural sizing already
qualifies; `xorfuse` pays 2× bits/key to get there; `batched` /
`krassovsky` **cannot** hit 0.001 at any sizing because their fixed
K=4 / K=5 floors the achievable FP):

| Implementation        | bits/key  | actual FP | prehash miss S | prehash miss M |
|-----------------------|----------:|----------:|---------------:|---------------:|
| **quickbloom: single_key** | 21.0 |   0.00034 |       **1.26** |           2.90 |
| quickbloom: unified   |      21.0 |   0.00034 |           1.47 |       **2.87** |
| quickbloom: batched   |      21.0 |   0.00257 ⚠ |           1.93 |           3.18 |
| `krassovsky`          |      21.0 |   0.00346 ⚠ |           4.69 |           5.67 |
| `xorfuse`             | **41.9** ↑ |   0.00033 |           4.16 |           4.77 |
| `impala`              |      21.0 |   0.00034 |           8.12 |          10.27 |
| `classic`             |      21.0 |   0.00016 |           7.75 |          11.08 |
| `fastbloom`           |      21.0 |   0.00009 |           9.77 |          14.05 |

⚠ Couldn't hit 0.001 — K-floor of the algorithm prevents it.
↑ Resized: `xorfuse` doubled bits/key to satisfy the target.

### Notes

- **`single-key`** API is unconstrained — works for caches,
  streaming dedup, point lookups. **`batched-only`** (`krassovsky`)
  requires keys delivered in groups of 8 and isn't applicable to
  per-key callers. **`static set`** (`xorfuse`) needs the full key
  set up-front (~25–210 ns/key build vs ~2–7 ns/key for SBBF).
- **Where the gap comes from** (prehash, all hashing factored out):
    - Against `classic`: SBBF probes one cache line; `classic` does
      K=8 scattered probes. 6–7× gap at S/M, wider at L.
    - Against `impala` / `arrow_rs`: same SBBF block geometry (one
      cache line per probe), but their bit-tests are scalar where
      `single_key` uses SIMD mask compute (`vpmullo` + `vpsllv` +
      `_mm256_testc_si256`). ~6× gap.
    - Against `fastbloom`: a non-SBBF multi-block layout that
      touches more memory per probe than a 256-bit SBBF block.
    - Against `krassovsky`: matches SBBF's one-load-per-probe via
      `vpgatherqq` (4 blocks per instruction); the only algorithmic
      peer — when batched.
- **Reproduce**: `CC=clang python3 bench_all.py --sizes S,M,L
  --comparisons` (main tables); add `--target-fp 0.001` for the
  equal-FP variant; `make bench-hash` for the per-hash table.

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
