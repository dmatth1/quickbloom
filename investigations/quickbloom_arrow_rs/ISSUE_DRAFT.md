# Issue draft for `apache/arrow-rs`

Ready-to-paste GitHub issue body. File at https://github.com/apache/arrow-rs/issues/new.

**Suggested title:** Proposal: SIMD-accelerate the Parquet bloom filter probe (3× on cache-resident, 2× in DRAM)

---

## Summary

The `parquet::bloom_filter::Sbbf` probe is currently a scalar `for i in 0..8` loop over 32-bit lanes. Each Parquet block is 256 bits (8 × `u32`) — exactly an AVX2 vector — and the K=8 bit-tests reduce to a single `vptest` instruction on x86. Same pattern on AArch64 / NEON.

I have a reference Rust implementation that drops in alongside the existing scalar code with a runtime AVX2 dispatch + scalar fallback. Same hash, same wire format, same algorithm, same `Sbbf::check<T: AsBytes>` / `Sbbf::insert<T: AsBytes>` API. **Measured 2.78–3.27× speedup on miss-heavy probes across cache regimes** on a same-host A/B against the current scalar code.

Looking for maintainer signal before I open the PR. Questions below.

## Motivation

The Parquet bloom filter is on the hot path of every Parquet row-group skip in:

- **Polars** (`polars-parquet` → `arrow-rs`)
- **Databend**, **InfluxDB 3.0 / IOx**, **Quickwit**, **LanceDB**, **RisingWave**, **GreptimeDB**
- **DataFusion** and all engines built on it

A SIMD probe propagates to all of them automatically — no API change, no version bump on their side, just an arrow-rs minor version.

## Current state

`parquet/src/bloom_filter/mod.rs` lines ~210–230 (current `main`):

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

fn insert(&mut self, hash: u32) {
    let mask = Self::mask(hash);
    for i in 0..8 {
        self[i] |= mask[i];
    }
}
```

These are correct and the per-block layout matches the Parquet SBBF spec. The opportunity is purely the SIMD probe — the scalar loop can be replaced with one `_mm256_testc_si256` reduction.

## Proposed change

Add `block_check_avx2` and `block_insert_avx2` using `std::arch::x86_64` (stable since 1.27), plus dispatch wrappers that check `std::is_x86_feature_detected!("avx2")` and fall back to the existing scalar code otherwise. ~100 LOC total in one file.

Same shape as the Apache Arrow C++ xsimd port that's currently in `[DISCUSS]` on `dev@arrow.apache.org` ([thread link]). Same algorithm and on-disk format on both sides; only the inner SIMD operations differ.

## Measured speedup

Reference implementation: `dmatth1/quickbloom @ quickbloom-arrow-rs`, file `comparisons/quickbloom_arrow_rs_shim/src/lib.rs` — a `cdylib` exposing the same C ABI as the existing arrow-rs scalar shim, so the same bench harness measures both. Captured S/M/L runs in `investigations/quickbloom_arrow_rs/results/run5_post_hash_fix.txt` and `run6_lxl_post_fix.txt`.

Host: Intel Xeon @ 2.1 GHz (Sapphire Rapids, virtualised), clang -O3, rustc 1.94. Contains-miss hash+bloom path, median ns/op:

| Regime | arrow-rs `Sbbf` (current, scalar) | Reference SIMD port | Speedup |
|--------|----------------------------------:|--------------------:|--------:|
| S  (128 KB filter, in L2) | 18.52 | **5.66** | **3.27×** |
| M  (2 MB filter, in L3) | 23.75 | **8.55** | **2.78×** |
| L  (32 MB filter, around L3) | 30.59 | **12.53** | **2.44×** |

Both sides use the same hash (`XxHash64::oneshot` from twox-hash 2.x, same as `Sbbf`); the only difference is the probe path. Insert path is similar: 1.06× in L2, 1.16× in M, 1.17× in L (the K=8 OR-store becomes one `vpor` instead of 8 scalar ORs).

A "probe-only" comparison (no hashing) would show ~8× — but arrow-rs's `check_hash` / `insert_hash` are private, so I can't bench it via the public API. If they were made `pub`, callers that batch-hash columns (a common pattern in analytical engines) would get the full SIMD win amortised. That's an orthogonal follow-up, not part of this proposal.

## Questions for maintainers

1. **Portability strategy**. I'd prefer `std::arch::x86_64` for the AVX2 path (matches arrow-rs's existing SIMD pattern in `arrow-arith/`, `arrow-buffer/`, etc.) with scalar fallback elsewhere. NEON path as a follow-up PR. Alternative: write once using the `wide` crate (~5% perf cost, +1 dep). Preference?
2. **MSRV**. `std::arch::x86_64` has been stable since 1.27, and `std::is_x86_feature_detected!` since 1.27 — no MSRV bump needed. Confirming this is fine?
3. **Bench harness**. There's no existing `parquet/benches/bloom_filter_check.rs`. Happy to add one. Should it match the style of `parquet/benches/arrow_writer.rs` or something else?
4. **Scope**. PR adds AVX2 + scalar fallback for `check` and `insert`. Not touching: bulk APIs, the public `Sbbf::check<T: AsBytes>` signature, the wire format, the hash, MSRV, or anything else. NEON is a follow-up; making `check_hash`/`insert_hash` public is a follow-up. Sound right?

If the above looks reasonable I'll prepare the PR. Reference branch is `dmatth1/quickbloom @ quickbloom-arrow-rs` if you want to look at the algorithm in advance.
