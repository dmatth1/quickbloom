# quickbloom

A fast Split Block Bloom Filter for x86_64 with AVX2. One
implementation: `wymum` hash + SIMD mask compute, single 256-bit
cache-line block per probe. The fastest single-key SBBF probe kernel
on AVX2 x86_64 across the cache regimes we bench. See
[Performance](#performance) for the head-to-head.

Reference implementations of other Bloom designs (Apache Parquet/Impala
SBBF, Save Buffer PatternedSimd, textbook K-hash Bloom, binary fuse
filter, arrow-rs SBBF, fastbloom) live in
[`comparisons/`](comparisons/README.md) and are built by the same
harness when you pass `--comparisons`.

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

void* f = qb_new(qb_estimate_bits(100000, 0.01));   // 100k items, 1% FP
qb_insert(f, "hello", 5);
if (qb_contains(f, "hello", 5)) { /* probably present */ }
qb_free(f);
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

## Algorithm

Split Block Bloom Filter
([Apache Parquet spec](https://github.com/apache/parquet-format/blob/master/BloomFilter.md)):

- 256-bit blocks (8 × 32-bit words), K=8 (one bit per word). Every
  probe touches exactly one cache line.
- Spec-mandated salt vector, so the bitset is bit-identical to other
  Parquet SBBF implementations (arrow-cpp, arrow-rs, Velox, DuckDB,
  Impala) for the same 64-bit hash input.
- Power-of-2 block count with bitmask block-index (no `fastrange`).
- AVX2 SIMD mask compute: `vpmullo` + `vpsrli` + `vpsllv` + `vptest`.
- 4-way unrolled bulk paths. Insert uses sequential load-or-store per
  key so store-to-load forwarding handles aliased blocks correctly.

Hash is `wymum` — one 128-bit multiply on the 16-byte fast path — with
a `fasthash64` fallback for variable-length keys.

## ABI

Full declarations are in `quickbloom.h`.

```c
// Lifecycle. qb_new returns NULL on alloc failure; qb_free accepts NULL.
void*  qb_new(size_t nbits);
void   qb_free(void* p);

// Single-key API.
void   qb_insert(void* p, const void* key, size_t len);
int    qb_contains(void* p, const void* key, size_t len);

// Bulk API. Returns the number of hits.
void   qb_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n);
size_t qb_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n);

// Pre-hashed (for callers that already have a 64-bit hash of the key).
void   qb_insert_prehash(void* p, uint64_t hash);
int    qb_contains_prehash(void* p, uint64_t hash);
void   qb_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n);
size_t qb_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n);

// Sizing helper. Returns recommended bit count for ~n items at FP rate fp.
size_t qb_estimate_bits(size_t n, double fp);
```

### Thread safety

- Concurrent **reads** (`qb_contains*`) are safe.
- Concurrent **writes** (`qb_insert*`) are **not** safe (non-atomic
  bit-set). Callers must synchronize. (Same convention as arrow-rs,
  impala, parquet-rs.)
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

> Numbers are illustrative and **host-dependent**. Run `make bench` on
> your hardware. Captured here on Intel Xeon @ 2.8 GHz (Cascade Lake-
> class, 33 MB L3), clang -O3, contains-miss, min ns/op.

### quickbloom

| Size           | hash+bloom miss | prehash miss |
|----------------|----------------:|-------------:|
| S (128 KB, in L2) | 1.95         | 1.44         |
| M (2 MB, in L3)   | 9.28         | 5.01         |
| L (32 MB, around L3) | 16.09     | 12.04        |

Hash+bloom is what a user calling `qb_contains(bytes, len)` pays end
to end (`wymum` hash + SBBF probe). Prehash is the kernel cost alone,
for callers passing pre-computed `uint64` hashes via
`qb_contains_prehash`.

### Comparison vs other libraries

Same prehash metric, so we're comparing probe kernels (hashing
factored out on both sides):

| Implementation         | K | S       | M       | L       | FP rate |
|------------------------|---|--------:|--------:|--------:|--------:|
| **quickbloom**         | 8 | **1.44**| **5.01**| **12.04**| 0.00030 |
| `xorfuse` (binary fuse)| 3 |    4.52 |    7.38 |    23.53| 0.00422 |
| `krassovsky` ¹         | 5 |    4.85 |    7.66 |    15.79| 0.00340 |
| `classic`              | 8 |    8.41 |   13.07 |    23.45| 0.00013 |
| `impala`               | 8 |    8.54 |   10.96 |    19.87| 0.00030 |
| `fastbloom` (Rust)     | 8 |   10.65 |   17.12 |    37.10| 0.00006 |
| `arrow_rs` SBBF ²      | 8 |       — |       — |        —| 0.00030 |

¹ **`krassovsky` (Save Buffer `PatternedSimd`) isn't directly
comparable.** It's a batched-only design — every `contains` issues a
`vpgatherqq` that loads 4 blocks per instruction, with no
comparably-tuned single-key path. The numbers above are measured
through its 8-way `bloom_*_batch8_bulk` entry points; for per-key
workloads (the common case for caches, point lookups, streaming
dedup) the design doesn't apply. K=5 also means ~10× higher FP than
quickbloom at equal sizing.

² `arrow_rs` (the parquet crate) exposes no `insert_hash` /
`check_hash`. Its hash+bloom numbers are 21.52 / 29.18 / 74.65 ns at
S/M/L; structurally it's the same algorithm as `impala` (SBBF +
XXH64), and its kernel would land near `impala`'s.

### Per-key hash cost

Kernel cost on 16-byte keys, same compile flags as the bloom bench
(reproduce with `make bench-hash`): `wymum16` 0.90 ns/op
(quickbloom), `XXH64` 2.80 ns/op (impala, arrow_rs), `SipHash-1-3`
3.40 ns/op (fastbloom). Lets you reconstruct any candidate's
hash+bloom from its prehash number. Real-world `fastbloom`
end-to-end hash overhead is closer to ~15 ns because the Rust
`Hasher` trait adds per-call dispatch and the seed is loaded from
filter state on every call.

### Where the prehash gap comes from

- Vs `classic`: SBBF probes one cache line; `classic` does K=8
  scattered probes.
- Vs `impala` / `arrow_rs`: same SBBF block geometry, but scalar
  bit-test vs SIMD mask compute (`vpmullo` + `vpsllv` +
  `_mm256_testc_si256`).
- Vs `fastbloom`: a non-SBBF multi-block layout that touches more
  memory per probe than a 256-bit SBBF block.
- Vs `xorfuse`: different class (static set, ~9 bits/key, 3 cache
  lines per probe). Loses on probe latency, wins on memory.

Reproduce: `CC=clang python3 bench_all.py --sizes S,M,L --comparisons`.

## Optimizations we tried and rejected

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

## Files

```
quickbloom.h         public header
quickbloom.c         the SBBF implementation
qb_util.c            qb_estimate_bits
Makefile             build libquickbloom.{a,so} + test + example; supports install
quickbloom.pc.in     pkg-config template; installed as quickbloom.pc
examples/            minimal usage example
test/                native C correctness tests
tools/bench_hash.c   per-hash kernel-cost bench (make bench-hash)
.github/workflows/   CI: build + test on gcc and clang
harness.py           compile + diff_test + benchmark infrastructure
bench_all.py         benchmark sweep across cache regimes; --target-fp / --comparisons
test_bloom.py        Python correctness tests (fixed-length and variable-length keys)
comparisons/         reference Bloom implementations (impala, krassovsky, classic,
                     xorfuse, fastbloom, arrow-rs) plus their own README
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
