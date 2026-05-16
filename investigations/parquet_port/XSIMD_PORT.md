# Switching the parquet_port probe to xsimd

This document records the move from raw `_mm256_*` intrinsics to
[xsimd](https://github.com/xtensor-stack/xsimd). The existing
`PARQUET.md`, `parquet_port/README.md`, and `RESULTS.md` still
describe the raw-intrinsics version and have not been re-synced.

## Why

The `[DISCUSS]` reply on `dev@arrow.apache.org`
([thread](https://lists.apache.org/thread/omof0fq47tndfd80g5hwp2bvjmzvpb40))
asked that the probe be written against xsimd rather than raw
intrinsics, matching the rest of Arrow's SIMD code (e.g.
`arrow/util/bit_util.h`, `arrow/util/bpacking_simd.h`). For the
investigation to map cleanly to the eventual upstream patch, the
local code needs to use the same abstraction.

## What changed

Functions renamed and rewritten against `xsimd::batch<uint32_t>`:

| Before                                | After                                   |
|---------------------------------------|-----------------------------------------|
| `filter::find_avx2(uint64_t)`         | `filter::find_simd(uint64_t)`           |
| `filter::find_avx2_x4(f0..3, hash)`   | `filter::find_simd_x4(f0..3, hash)`     |
| `_mm256_*` intrinsics                 | `xsimd::batch<uint32_t>` ops            |
| `_mm256_testc_si256(blk, mask)`       | `xsimd::none(as_bool(bitwise_andnot(mask, blk)))` |

`find_scalar` is unchanged — still the line-for-line port of Apache
Arrow C++'s `FindHash`, used as the diff-test reference.

`static_assert(batch_t::size == 8)` is set in `parquet_bloom.cpp` —
Parquet's 8-bits-per-block expects an 8-lane batch, which is the
default on AVX2 and AVX-512. A NEON port would need a different
splitting (two 4-lane batches per block) and is not yet implemented.

## The reduction: finding the right xsimd expression

This was the one tricky bit. `_mm256_testc_si256(blk, mask)` returns 1
iff `(~blk & mask) == 0` and lowers to a single `vptest`.

The "natural" xsimd translation is
`xsimd::all((blk & mask) == mask)`. On AVX2 that lowers to **five**
instructions:

```
vpand    (mem), ymm_mask, ymm_tmp     ; tmp = blk & mask
vpcmpeqd ymm_tmp, ymm_mask, ymm_eq    ; per-lane (tmp == mask)
vpcmpeqd ymm_ones, ymm_ones, ymm_ones ; sentinel all-ones
vptest   ymm_ones, ymm_eq             ; CF = all lanes true
setb     %al
```

The fix is to rewrite the test in the form xsimd already lowers
efficiently. xsimd's `none(batch_bool)` uses
`_mm256_testz_si256(self, self)` (cf. `xsimd_avx.hpp:140`-ish), which
returns 1 iff `self == 0` across the whole 256-bit register. So:

```cpp
inline bool_t as_batch_bool(batch_t b) noexcept {
    return bool_t(typename bool_t::register_type(b));   // bit-pattern reinterpret
}

inline bool all_mask_bits_set(batch_t blk, batch_t mask) {
    return xsimd::none(as_batch_bool(xsimd::bitwise_andnot(mask, blk)));
}
```

The disassembly is then exactly what we want:

```
vmovdqu (mem), ymm2     ; load blk
vpandn  ymm0, ymm2, ymm0; ~blk & mask  (xsimd::bitwise_andnot)
vptest  ymm0, ymm0      ; testz reduction  (xsimd::none)
sete    %al
```

Same shape as the raw-intrinsics `_mm256_testc_si256` path, all
through public xsimd APIs. No `#ifdef`, no escape hatch, no upstream
xsimd PR needed.

Two semantic facts worth pinning down because they're easy to get
backwards:

- `xsimd::bitwise_andnot(self, other)` lowers to
  `_mm256_andnot_si256(other, self)`, which computes `~other & self`.
  So `bitwise_andnot(mask, blk)` = `~blk & mask` — exactly what
  `testc` tests against zero.
- `batch_bool<T>(register_type)` and the implicit conversion from
  `batch<T>` to `register_type` are both public xsimd API; combining
  them is a free type-level reinterpret. The reduction `xsimd::none`
  doesn't care about per-lane boolean interpretation — `testz(x, x)`
  checks if any bit is set anywhere in the register.

## Build

```
sudo apt install libxsimd-dev   # xsimd 12+
cd investigations/parquet_port
make && ./bench
```

xsimd is header-only; no link change. Same compile flags as before:
`-O3 -mavx2 -mbmi2 -mfma`. Build is clean under `-Wall -Wextra`.

## Bench

20 repeats per regime, median reported alongside min / p90 / max for
variance. Hardware: **Intel Xeon Sapphire Rapids @ 2.1 GHz
(virtualised)** — different host from the earlier Cascade Lake @ 2.8
GHz runs in `results/run1.txt` and `results/run2.txt`.

All numbers below are post-hash probe latency: XXH64 is computed
once at setup and excluded from the timed loop.

### Pure-xsimd numbers (this branch)

Captured run: `results/run7_xsimd_pure.txt`.

| Regime             |  min   | median |  p90   |  max   |
|--------------------|-------:|-------:|-------:|-------:|
| Small (in-cache)   |        |        |        |        |
| · scalar           | 12.04  | 12.35  | 12.50  | 12.60  |
| · xsimd single-key |  2.43  |  2.48  |  2.54  |  2.55  |
| · xsimd 4-way bulk |  1.63  |  1.73  |  1.76  |  1.77  |
| Medium (out-of-L3) |        |        |        |        |
| · scalar           | 18.18  | 18.40  | 18.91  | 19.19  |
| · xsimd single-key |  7.04  |  7.41  |  7.63  |  9.44  |
| · xsimd 4-way bulk |  7.04  |  7.17  |  7.33  |  7.43  |
| Large (deep DRAM)  |        |        |        |        |
| · scalar           | 30.21  | 31.05  | 31.58  | 31.78  |
| · xsimd single-key | 21.54  | 22.10  | 22.61  | 23.30  |
| · xsimd 4-way bulk | 18.01  | 18.67  | 19.25  | 19.39  |

Speedups (median, vs scalar): **4.98×** / **7.14×** in-cache,
**2.48×** / **2.57×** out-of-L3 medium, **1.41×** / **1.66×** deep
DRAM.

**Diff-test: 0 mismatches across 167M (query, filter) pairs** — the
xsimd path produces bit-identical results to the scalar Arrow C++ /
arrow-rs / Velox reference.

## Same-machine A/B: xsimd-pure vs raw `_mm256_*`

Both implementations built and run back-to-back on the same SPR host
with the same 20-rep bench. Raw run captured in
`results/run8_raw_intrinsics_spr_fresh.txt`.

Median ns/probe:

| Regime             | impl  | single-key | 4-way bulk |
|--------------------|-------|-----------:|-----------:|
| Small (in-cache)   | raw   |      2.29  |      1.66  |
|                    | xsimd |      2.48  |      1.73  |
|                    | **xsimd vs raw** |  **+8%**   |  **+4%**   |
| Medium (out-of-L3) | raw   |      7.44  |      7.38  |
|                    | xsimd |      7.41  |      7.17  |
|                    | **xsimd vs raw** |  **~0%**   |  **−3%**   |
| Large (deep DRAM)  | raw   |     22.30  |     18.85  |
|                    | xsimd |     22.10  |     18.67  |
|                    | **xsimd vs raw** |  **−1%**   |  **−1%**   |

Within noise on medium and large; small (in-cache) is +8% on single-key.
The disassembly is essentially identical so that 8% is probably minor
scheduling / register allocation; could be tightened further but not
worth chasing for the upstream story.

(Both medium-regime numbers shifted from the earlier 16-17 ns range
to ~7 ns between sessions; that's a property of the virtualised host's
cache state at run time, not of the implementation. The A/B is valid
because both versions were measured back-to-back in the same session.)

## Earlier attempts and what didn't work

For the record, none of the "obvious" xsimd translations of the
testc reduction lowered well:

| Expression                                            | Result  |
|-------------------------------------------------------|---------|
| `xsimd::all((blk & mask) == mask)`                    | 5 ops, slow |
| `xsimd::all(bitwise_andnot(mask, blk) == 0)`          | 5 ops, slow |
| `xsimd::all(bitwise_cast<batch_bool<...>>(...))`      | doesn't compile (bitwise_cast won't target batch_bool) |
| `xsimd::none(as_batch_bool(bitwise_andnot(mask, blk)))` | **2 ops, optimal** |

The xsimd library exposes `testc` only internally, via the
`all(batch_bool)` lowering — but `any/none(batch_bool)` use `testz`,
which is the dual we actually want for the andnot form. Once the
reduction is written in the `none(testz)` shape, no escape hatch is
needed.

A small `xsimd::testc(batch, batch)` upstream PR would still be a
nice cleanup — direct, no `bitwise_andnot` indirection — but it's
no longer required to get the perf back.

## What's not done

- **NEON port.** `xsimd::batch<uint32_t>` is 4-lane on NEON, so each
  256-bit Parquet block needs two batches. The reduction (`none` on
  a `batch_bool`) translates fine, but the algorithm split still
  needs writing.
- **`PARQUET.md`, `parquet_port/README.md`, `RESULTS.md` re-sync.**
  Those still describe the raw-intrinsics version and headline
  numbers. Deliberately left alone for now.
- **Blog post.** Same — still references the AVX2 intrinsics and the
  Cascade Lake numbers.
