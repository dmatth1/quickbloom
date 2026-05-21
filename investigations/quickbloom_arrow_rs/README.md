# quickbloom-arrow-rs

Investigation porting quickbloom's SBBF (`bloom_single_key.c` /
`bloom_unified.c`) to Rust and measuring head-to-head against
arrow-rs's `parquet::bloom_filter::Sbbf` — the production Rust
Parquet bloom filter, which is fully scalar.

## What this is

arrow-rs's `Sbbf::check` and `Sbbf::insert` (verified
2026-05-19 by reading `parquet/src/bloom_filter/mod.rs`) are
pure scalar:

```rust
fn check(&self, hash: u32) -> bool {
    let mask = Self::mask(hash);
    for i in 0..8 {
        if self[i] & mask[i] == 0 {
            return false;
        }
    }
    true
}
```

No SIMD intrinsics, no `std::simd`, no bulk-probe API. Same
algorithm shape as arrow-cpp's scalar SBBF before our
`parquet-xsimd-port` patch — just in Rust.

The Pattern B opportunity is direct: SIMD probe + bulk +
prefetch, mirroring the wins we already shipped for arrow-cpp.

## 1:1 with quickbloom C — what's ported

Verified line-by-line against `/home/user/quickbloom/bloom_sbbf.c`.

| Optimization trick | quickbloom C | Rust port (`quickbloom_arrow_rs_shim`) |
|---|---|---|
| 256-bit blocks, K=8 lane-aligned | ✅ | ✅ |
| Parquet SBBF SALT constants | ✅ | ✅ same 8 u32s |
| Power-of-2 nblocks + bitmask block index | `((h>>32) & nblocks_mask)` | `((h>>32) & block_mask) as usize * 8` |
| 32-byte aligned heap allocation | `posix_memalign(.., 32, ..)` | `std::alloc::Layout::from_size_align(.., 32)` |
| Aligned SIMD load/store | `_mm256_load_si256` / `_mm256_store_si256` | same intrinsics (`load_si256`, not `loadu_si256`) |
| SIMD mask compute | `vpmulld + vpsrld(27) + vpsllvd` | `_mm256_mullo_epi32` + `_mm256_srli_epi32::<27>` + `_mm256_sllv_epi32` |
| SIMD reduce | `_mm256_testc_si256` | `_mm256_testc_si256` |
| Single-key Add: load+OR+store | `_mm256_or_si256` | `_mm256_or_si256` |
| 4-way bulk insert with `APPLY` macro | ✅ | ✅ same 4-way load+OR+store unroll |
| 4-way bulk contains | ✅ | ✅ same 4-way load+testc unroll |
| Prefetch lookahead in bulk | `__builtin_prefetch(.., 0, 0)` (PREFETCH_LOOKAHEAD=8) | `_mm_prefetch::<_MM_HINT_T0>(..)` (PREFETCH_LOOKAHEAD=8) |
| Scalar fallback for non-AVX2 | ✅ | ✅ |

**Intentional difference from quickbloom C:** hash function.
Quickbloom uses wymum-style `hash16` (16-byte fast path) +
`fasthash64_var`. The Rust port uses **XXH64** (matches arrow-rs's
Sbbf and the Parquet spec) so the head-to-head measures the
*SBBF algorithm + SIMD tricks* delta, not a hash difference.
The bench's prehash columns isolate the algorithm-only number
that's directly comparable to quickbloom C.

## Files

- `../../comparisons/quickbloom_arrow_rs_shim/` — the Rust port,
  built as a cdylib (`crate-type = ["cdylib"]`) so the bench
  harness can `dlopen` it via the same `bloom_*` C ABI as the
  other comparison candidates.
- `../../comparisons/arrow_rs_sbbf_shim/` — the existing shim
  for arrow-rs's `Sbbf` itself (production, scalar).
- `../../bench_all.py` — both shims registered as comparison
  candidates.
- `results/run2.txt` — the head-to-head bench output (when the
  bench finishes).

## Reproducing

```sh
cd ..  # repo root
CC=clang python3 bench_all.py --sizes S,M,L,XL --comparisons
```

The relevant rows: `arrow_rs_sbbf_shim` (production Rust SBBF,
scalar) vs `quickbloom_arrow_rs_shim` (our 1:1 SIMD port) vs
`single_key` (the C reference — `quickbloom_arrow_rs_shim`
should be within striking distance of this in cache-resident
regimes, and tied in DRAM).

## What's not yet done

- Bench against an actual `cargo bench`-style harness inside
  arrow-rs's own test suite (would be the eventual PR validation
  step).
- NEON port for AArch64. Scalar fallback runs there; no SIMD.
- File the actual arrow-rs PR. This branch is the reference
  implementation + bench evidence; PR filing is a separate step.

## Honest caveats

1. Our Rust port reproduces quickbloom's optimisation tricks but
   it's not the same binary as quickbloom's C reference — it goes
   through Rust's codegen + the function-pointer dispatch we use
   for `is_x86_feature_detected!`. Numbers may be slightly different
   from the C single_key for that reason.
2. Single host (Sapphire Rapids @ 2.1 GHz, virtualised). The bench
   reproduces on any host with AVX2.
3. The hash choice (XXH64) is deliberate to match arrow-rs's
   `Sbbf`. A real arrow-rs PR would keep XXH64 (their spec
   choice); this isn't a free knob to swap.
