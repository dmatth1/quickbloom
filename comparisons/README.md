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

Intel Xeon @ 2.8 GHz (Cascade Lake-class, 33 MB L3), clang -O3, S =
128 KB filter, contains-miss, ns/op min:

| Implementation                | K | hash+bloom | prehash | cyc/op (prehash) | FP rate |
|-------------------------------|---|-----------:|--------:|-----------------:|--------:|
| `single_key` (top-level)      | 8 |   **1.99** |**1.44** |          **4.0** | 0.00034 |
| `unified` (top-level)         | 8 |       2.56 |    1.74 |              4.9 | 0.00034 |
| `batched` (top-level)         | 4 |       2.72 |    2.09 |              5.9 | 0.00257 |
| `bloom_krassovsky.c`          | 5 |       5.63 |    4.96 |             13.9 | 0.00346 |
| `bloom_classic.c`             | 8 |       9.40 |    8.49 |             23.8 | 0.00016 |
| `bloom_impala.c`              | 8 |      17.56 |    8.51 |             23.8 | 0.00034 |

At L (32 MB, around L3):

| Implementation                | hash+bloom miss (ns/op) | prehash miss (ns/op) |
|-------------------------------|------------------------:|---------------------:|
| `unified` (top-level)         |                **17.83**|             **11.61**|
| `single_key` (top-level)      |                   20.05 |                14.33 |
| `krassovsky`                  |                   22.50 |                21.05 |
| `classic`                     |                   24.25 |                17.89 |
| `batched` (top-level)         |                   26.99 |                13.27 |
| `impala`                      |                   57.40 |                22.43 |

## Caveats

* **Gather is CPU-dependent.** Krassovsky uses `vpgatherqq` for the
  contains path. On older gather implementations (Cascade Lake)
  `single_key` is ~3× faster; on Sapphire Rapids (improved gather
  latency) the gap narrows to ~16%. Either way `single_key` wins, but
  the margin shifts by CPU.

* **Krassovsky's K is approximate.** PatternedSimd's mask table sets
  4–5 bits per mask with a sliding window; the parser sees K=5. The
  realized FP rate is ~8× the theoretical reported in the K=5 column
  because the algorithm is not a true independent-k-position Bloom.

* **Impala's gap is half hash, half algorithm.** Compared to
  `single_key`'s wymum, XXH64 is several times slower per key. In
  prehash mode (hash excluded from the timed loop) Impala lands at
  ~8.5 ns/op — roughly the same as `bloom_classic.c`. The remaining
  algorithmic gap is "scalar bit-set vs SIMD mask compute."

* **`bloom_classic.c` uses double-hashing** (Kirsch–Mitzenmacher) to
  cheaply derive K positions from one 128-bit multiply. A "more
  textbook" version would compute K truly independent hashes; that
  would be slower again.

* **fastbloom (Rust)** is not ported here. The ~3–5 ns/op figure cited
  in the literature is from the crate's own benchmarks on different
  hardware. Adding a Rust port to this bench is a follow-up.
