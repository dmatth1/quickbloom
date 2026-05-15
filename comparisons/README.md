# Reference comparisons

External Bloom filter implementations included here for direct,
head-to-head measurement against the three designs at the top of
this repo. Each file is built and benchmarked by the same harness
that builds the main designs — these are not asserted numbers from
an upstream README, they are measured on whatever host you run on.

To include them in a bench run:

```sh
CC=clang python3 ../bench_all.py --comparisons
```

To run only correctness tests against them:

```sh
python3 ../test_bloom.py
```

## What's here

| File | Source | Algorithm |
|---|---|---|
| `bloom_impala.c` | parquet-format/BloomFilter.md | SBBF, 256-bit blocks, K=8, Parquet salts, **scalar bit-set**, **XXH64** hash |
| `bloom_krassovsky.c` | [save-buffer/bloomfilter_benchmarks](https://github.com/save-buffer/bloomfilter_benchmarks) (MIT) | PatternedSimd: 64-bit blocks, 1024-entry mask table (4–5 bits set, sliding window), AVX2 8-way batched, `vpgatherqq` contains |
| `bloom_classic.c` | textbook | K=8 double-hashing (`h_i = h1 + i·h2`), non-blocked. Pre-SBBF baseline. |

## Measured head-to-head

Intel Xeon Sapphire Rapids @ 2.1 GHz, clang -O3, S = 128 KB filter,
contains-miss (hash+bloom path), ns/op min:

| Implementation               | K | ns/op | cyc/op | FP rate |
|------------------------------|---|------:|-------:|--------:|
| `single_key` (top-level)     | 8 | **1.62** | **3.4** | 0.00034 |
| `unified` (top-level)        | 8 |  2.03 |   4.3  | 0.00034 |
| `batched` (top-level)        | 4 |  2.84 |   6.0  | 0.00257 |
| `bloom_krassovsky.c`         | 5 |  1.87 |   3.9  | 0.00346 |
| `bloom_impala.c`             | 8 | 17.53 |  36.8  | 0.00034 |
| `bloom_classic.c`            | 8 | 10.33 |  21.7  | 0.00016 |

Out of L3 (L = 32 MB, contains-miss), `batched` (8.68 ns) beats
`krassovsky` (9.11 ns) — direct confirmation that scalar 8-byte
loads beat `vpgatherqq` on this CPU.

## Caveats

* **Krassovsky's K is approximate.** PatternedSimd's mask table sets
  4–5 bits per mask with a sliding window; the parser sees K=5. The
  realized FP rate is ~8× the theoretical reported in the K=5 column
  because the algorithm is not a true independent-k-position Bloom.

* **Impala uses XXH64.** Compared to `single_key`'s wymum (one 128-bit
  multiply), XXH64 is several times slower per key. The 10× gap to
  `single_key` is part hash, part algorithm — not pure SIMD-mask vs
  scalar-mask.

* **`bloom_classic.c` uses double-hashing** (Kirsch–Mitzenmacher) to
  cheaply derive K positions from one 128-bit multiply. A "more
  textbook" version would compute K truly independent hashes; that
  would be slower again.

* **fastbloom (Rust)** is not ported here. The ~3–5 ns/op figure cited
  in the literature is from the crate's own benchmarks on different
  hardware. Adding a Rust port to this bench is a follow-up.
