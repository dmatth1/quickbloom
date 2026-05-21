# quickbloom

Fast Split Block Bloom Filter for x86_64. wyhash-style hash +
AVX2 mask compute, one cache line per probe. The fastest single-key
SBBF kernel we've measured — see [Performance](#performance).

Reference Bloom implementations (impala, krassovsky, classic,
xorfuse, arrow-rs, fastbloom) live in
[`comparisons/`](comparisons/README.md).

## Quick start

```sh
git clone https://github.com/dmatth1/quickbloom
cd quickbloom
make                # build/libquickbloom.{a,so}
sudo make install   # /usr/local by default; override with PREFIX=
```

```c
#include <quickbloom.h>

void* f = qb_new(qb_estimate_bits(100000, 0.01));   // 100k items, 1% FP
qb_insert(f, "hello", 5);
if (qb_contains(f, "hello", 5)) { /* probably present */ }
qb_free(f);
```

Link with `-lquickbloom -lm`, or use `pkg-config --cflags --libs
quickbloom`. `examples/hello_quickbloom.c` is the full runnable
version (`make example`).

Tests and benches:

```sh
make test                                        # native C tests
python3 test_bloom.py                            # Python tests (incl. comparisons)
CC=clang python3 bench_all.py --comparisons      # bench sweep, all candidates
CC=clang python3 bench_all.py --sizes S,M        # subset
CC=clang python3 bench_all.py --target-fp 0.001  # equal-FP sizing
make bench-hash                                  # per-hash kernel cost
make bench-qb                                    # native C bench of the qb_* kernels
```

## Algorithm

Split Block Bloom Filter
([Apache Parquet spec](https://github.com/apache/parquet-format/blob/master/BloomFilter.md)):

- 256-bit blocks (8 × 32-bit words), K=8 (one bit per word). Every
  probe touches exactly one cache line.
- Spec-mandated salt vector, so the bitset is bit-identical to other
  Parquet SBBF implementations (arrow-cpp, arrow-rs, Velox, DuckDB,
  Impala) for the same 64-bit hash input.
- Power-of-2 block count. The Parquet-mandated fastrange block
  index `((h >> 32) * nblocks) >> 32` collapses to a single right
  shift of the upper hash half, so we get spec compliance at the
  same one-cycle cost as a bitmask.
- AVX2 SIMD mask compute: `vpmullo` + `vpsrli` + `vpsllv` + `vptest`.
- 4-way unrolled bulk paths. Insert uses sequential load-or-store per
  key so store-to-load forwarding handles aliased blocks.
- Hash on the 16-byte fast path is a single 128-bit multiply with
  xor-fold of the halves (wyhash-style, seeded so structured zero
  inputs don't collapse). Variable-length keys use `fasthash64`.

## ABI

Full declarations are in `quickbloom.h`.

```c
// Lifecycle. qb_new returns NULL on alloc failure; qb_free accepts NULL.
void*  qb_new(size_t nbits);
void   qb_free(void* p);

// Single-key API.
void   qb_insert(void* p, const void* key, size_t len);
int    qb_contains(void* p, const void* key, size_t len);

// Bulk API. qb_contains_bulk returns the number of hits.
void   qb_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n);
size_t qb_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n);

// Pre-hashed (skip the built-in wymum step).
void   qb_insert_prehash(void* p, uint64_t hash);
int    qb_contains_prehash(void* p, uint64_t hash);
void   qb_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n);
size_t qb_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n);

// Sizing helper. Returns recommended bit count for ~n items at FP rate fp.
size_t qb_estimate_bits(size_t n, double fp);

// Serialization. Raw bitset bytes, bit-identical to the Parquet
// on-disk layout (arrow-rs / arrow-cpp / Velox / DuckDB / Impala).
size_t qb_serialized_size(void* p);
void   qb_serialize(void* p, uint8_t* dst);
void*  qb_deserialize(const uint8_t* bytes, size_t nbytes);
```

### Thread safety

- Concurrent **reads** (`qb_contains*`) are safe.
- Concurrent **writes** (`qb_insert*`) are **not** safe (non-atomic
  bit-set). Callers must synchronize. (Same convention as arrow-rs,
  impala, parquet-rs.)
- Reads concurrent with writes may see partial state but never a false
  negative once the write completes.

## Build target

Built with `-O3 -mavx2 -mbmi2 -mfma -maes`. Runs on any x86_64 with
AVX2 + BMI2: Intel Haswell (2013)+ or AMD Excavator (2015)+ / Zen
1–5. AArch64 / Apple Silicon support is not planned — the bitset
layout is portable but the SIMD mask compute is x86-specific and
would need a NEON re-derivation.

## Performance

> Numbers are illustrative and **host-dependent**. Run `make bench`
> on your hardware. Captured on Intel Xeon @ 2.8 GHz (Cascade
> Lake-class, 1 MB L2 per core, 33 MB L3), clang -O3, contains-miss,
> min ns/op. Sizes span the cache hierarchy: S fits in L2, M and L
> spill to L3.

### quickbloom

Native C bench (`make bench-qb`), min ns/op across isolated runs:

| Size                 | hash+bloom | prehash |
|----------------------|-----------:|--------:|
| S (128 KB)           |       2.29 |    1.37 |
| M (2 MB)             |       5.73 |    2.70 |
| L (32 MB)            |      18.61 |   11.66 |

Hash+bloom is the full `qb_contains(bytes, len)` path. Prehash is
the kernel alone, for callers using `qb_contains_prehash(uint64)`.
The S→M jump reflects this host's L2/L3 boundary at 1 MB per core;
on hosts with larger L2 (Sapphire Rapids and newer have 2–8 MB per
core) M stays L2-resident and the jump shrinks.

### Comparison

Prehash mode, so probe kernels only:

| Implementation         | K | S       | M       | L       | FP rate |
|------------------------|---|--------:|--------:|--------:|--------:|
| **quickbloom**         | 8 | **1.37**| **2.70**| **11.66**| 0.00037 |
| `xorfuse` (binary fuse)| 3 |    4.52 |    8.49 |    25.24| 0.00394 |
| `krassovsky` ¹         | 5 |    4.96 |    7.11 |    25.15| 0.00353 |
| `classic`              | 8 |    8.57 |   12.74 |    32.11| 0.00013 |
| `impala`               | 8 |    8.63 |   11.11 |    24.18| 0.00032 |
| `fastbloom` (Rust)     | 8 |   10.57 |   17.30 |    44.34| 0.00011 |
| `arrow_rs` SBBF ²      | 8 |       — |       — |        —| 0.00032 |

¹ `krassovsky` isn't directly comparable: it's a batched-only
design (`vpgatherqq` loads 4 blocks per instruction; no
comparably-tuned single-key path). The numbers above are measured
through its 8-way `bloom_*_batch8_bulk` entry points. For per-key
workloads it doesn't apply. K=5 also gives ~10× higher FP than
quickbloom at equal sizing.

² `arrow_rs` (parquet crate) exposes no `insert_hash` /
`check_hash`. Hash+bloom numbers are 21.40 / 29.16 / 75.43 ns at
S/M/L; structurally the same SBBF + XXH64 as `impala`, so the
kernel would land near `impala`'s.

Where the prehash gap comes from:

- Vs `classic`: SBBF reads one cache line; `classic` does K=8
  scattered probes.
- Vs `impala` / `arrow_rs`: same SBBF block geometry, but scalar
  bit-test vs SIMD `vpmullo` + `vpsllv` + `_mm256_testc_si256`.
- Vs `fastbloom`: non-SBBF multi-block layout touches more memory
  per probe.
- Vs `xorfuse`: different class — static set, ~9 bits/key, 3 cache
  lines per probe. Loses on latency, wins on memory.

### Per-key hash cost

16-byte keys, same compile flags as the bloom bench, `make
bench-hash` (median of 5 isolated runs): `wymum16` 1.67 ns
(quickbloom), `XXH64` 3.50 ns (impala, arrow_rs), `SipHash-1-3`
3.38 ns (fastbloom). Numbers were re-captured on the current host
after the hash-robustness fix added a pre-multiply XOR pair;
`wymum16` is ~0.8 ns slower than before the fix, XXH64 sits a bit
above the prior capture (within typical host-to-host noise), and
SipHash is unchanged. Add to any candidate's prehash number for
hash+bloom latency. Real-world `fastbloom` overhead is closer to
~15 ns once the Rust `Hasher` trait dispatch and per-call seed
load are included.

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
| Bitmask block-index (skip fastrange) | failed bit-compat | `h & mask` reads the low bits of `h32`; Parquet's fastrange `(h32 * nblocks) >> 32` reads the high bits, so bitsets diverged from arrow-rs / arrow-cpp / Velox / DuckDB / Impala. Fastrange on power-of-2 nblocks collapses to a single right shift — same one-cycle cost, spec-compliant. |
| PGO (clang -fprofile-use) | <2% | code is mostly loop-free SIMD; PGO mostly helps branchy code |

## Methodology

Bench harness:

- Four sizes (`S/M/L/XL`) spanning L2 → DRAM, derived from the
  host's L2-per-core and L3 sizes at runtime (read from
  `/sys/devices/system/cpu/cpu0/cache/`; override with
  `QB_L2_KB` / `QB_L3_MB` env vars). Default sweep is `S,M,L`;
  `XL` needs `--sizes` and ~1 GB RAM.
- Random 16-byte keys, deterministic across runs (SHA-256 of an
  index, truncated). Separate seeds for inserted vs unseen sets.
- Repeats with warmup; reports min/median/p90. `min` is the
  headline (most reproducible).
- CPU, caches, and compiler are detected and printed in the
  bench header.
- The headline quickbloom numbers come from `make bench-qb`, a
  native C bench that times via `clock_gettime` inside the C
  binary (no ctypes boundary). The Python harness (`make bench`,
  `bench_all.py`) is the cross-candidate sweep and is what the
  comparison table uses; FFI overhead per bulk call is amortized
  over the call's n_keys and is below measurement noise.

Correctness (`test_bloom.py` and `diff_test` in the bench):

- Insert N keys, query them all back — zero false negatives.
- Query separately-seeded unseen keys — FP rate below 0.005.
- Variable-length keys (1–64 bytes) exercise the `fasthash64_var`
  path on every candidate.
- Structured-input regression suite (`test/test_quickbloom.c`):
  sequential u128, padded ASCII, UUIDv4-shaped — catches the
  zero-input collapse class of hash bugs.
- Bit-compat cross-validation (`test/test_compat_arrow_rs.py`):
  feeds the same XXH64 hash to a quickbloom filter (via the
  prehash API) and the arrow-rs `Sbbf` shim and asserts every
  probe agrees on a 50k-insert / 100k-query corpus. Verifies the
  Parquet bit-identical claim against the reference Rust
  implementation on every CI run.

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
tools/bench_qb.c     native C bench of the qb_* kernels (make bench-qb)
tools/fuzz_*.c       libFuzzer harnesses, incl. fuzz_deserialize (make fuzz)
tools/xxh64_helper.c standalone XXH64 .so for the cross-validation test
test/test_compat_arrow_rs.py   bit-compat cross-validation vs arrow-rs Sbbf
.github/workflows/   CI: build + test on gcc and clang
harness.py           compile + diff_test + benchmark infrastructure
bench_all.py         benchmark sweep across cache regimes; --target-fp / --comparisons
test_bloom.py        Python correctness tests (fixed-length and variable-length keys)
comparisons/         reference Bloom implementations (impala, krassovsky, classic,
                     xorfuse, fastbloom, arrow-rs) plus their own README
```

## Versioning

`0.1.0`. Semver on the C ABI in `quickbloom.h`; version macros
`QUICKBLOOM_VERSION_{MAJOR,MINOR,PATCH,STRING}` are exposed there.
Shared library uses standard SONAME (`libquickbloom.so.0` →
`libquickbloom.so.0.1.0`); `quickbloom.pc` ships for `pkg-config`.

## License

MIT — see [LICENSE](LICENSE).
