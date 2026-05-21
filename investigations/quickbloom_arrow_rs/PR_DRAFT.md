# PR draft for `apache/arrow-rs`

Ready-to-paste PR description. File at https://github.com/apache/arrow-rs/compare after forking + branching.

**Suggested title:** `parquet: SIMD-accelerate Sbbf probe (AVX2; 2.4–3.3× speedup)`

---

# Which issue does this PR close?

Closes #XXXX. (Replace with the proposal issue if you opened one — see `ISSUE_DRAFT.md` in the reference branch.)

# Rationale for this change

`Sbbf::check` and `Sbbf::insert` currently use a scalar `for i in 0..8` loop over the 8 × `u32` lanes of each 256-bit Parquet bloom block. The K=8 lane test reduces to a single AVX2 `vptest` instruction (`_mm256_testc_si256`), and the K=8 mask compute lowers to `vpmulld + vpsrld + vpsllvd`. This PR adds an AVX2 path with runtime feature detection, keeping the existing scalar code as the fallback for non-x86 and pre-AVX2 hosts.

The Parquet bloom filter is on the hot path of every row-group skip in Polars, Databend, InfluxDB 3.0 / IOx, Quickwit, LanceDB, RisingWave, GreptimeDB, and every DataFusion-based engine. They all benefit transparently from this change — no API break, no MSRV bump, no wire-format change.

# What changes are included in this PR?

Single file touched: `parquet/src/bloom_filter/mod.rs`.

- `block_check_avx2` (~12 LOC) and `block_insert_avx2` (~8 LOC) using `std::arch::x86_64`
- `mask_avx2` helper (~6 LOC) replacing the scalar `Self::mask` for the SIMD path
- Runtime dispatch in `check_hash` / `insert_hash`: `std::is_x86_feature_detected!("avx2")` → SIMD path; otherwise → existing scalar code
- Scalar functions preserved as the universal fallback (renamed slightly to `block_check_scalar` / `block_insert_scalar`)
- Tests verifying SIMD and scalar produce bit-identical output across 10K random hash inputs

Also adds `parquet/benches/bloom_filter_check.rs` with three regimes (small / medium / large filters, miss-heavy + hit-heavy workloads) to make the speedup reproducible by reviewers.

# Are there any user-facing changes?

No.

- `Sbbf::check<T: AsBytes>(&self, value: &T) -> bool` — same signature
- `Sbbf::insert<T: AsBytes>(&mut self, value: &T)` — same signature
- `Sbbf::check_hash` / `insert_hash` — same signature (still `pub(crate)`)
- On-disk format — unchanged (same Parquet SBBF spec, same XXH64 hash, same SALT constants)
- MSRV — unchanged (`std::arch::x86_64` and `is_x86_feature_detected!` have been stable since Rust 1.27)
- New dependencies — none

# Benchmark numbers

Same-host A/B on Intel Xeon @ 2.1 GHz (Sapphire Rapids), `rustc 1.94 -C target-cpu=native`, `--release`. Contains-miss path (the common case for Parquet row-group skip), median ns per probe:

| Regime | Filter size | Scalar (current) | AVX2 (this PR) | Speedup |
|--------|------------:|-----------------:|---------------:|--------:|
| Small  | 128 KB (in L2) |  18.52 |  **5.66** | **3.27×** |
| Medium | 2 MB (in L3) |  23.75 |  **8.55** | **2.78×** |
| Large  | 32 MB (around L3) |  30.59 | **12.53** | **2.44×** |

Insert path: 1.06×/1.16×/1.17× across the same regimes (the K=8 OR-store becomes one `vpor`).

Contains-hit path is roughly tied (both spend ~70% of the call inside `XxHash64::oneshot`). The SIMD probe win is fully visible on miss-heavy workloads, which is the dominant case for Parquet row-group skipping (most lookups expect to skip most row groups).

These numbers use the same `XxHash64::oneshot` path on both sides — only the probe inner loop changes.

Reference implementation + bench harness: `dmatth1/quickbloom @ quickbloom-arrow-rs`, `comparisons/quickbloom_arrow_rs_shim/`.

# Implementation notes

## Why `std::arch::x86_64` and not `std::simd` / `wide`

- `std::simd` is unstable (nightly only). MSRV considerations rule it out.
- `wide` crate would add a dependency and introduces ~5% perf overhead vs hand-rolled `std::arch` in this workload (the abstraction sometimes confuses LLVM's auto-vectorizer for the `vptest` reduction).
- `std::arch::x86_64` matches the existing SIMD pattern in arrow-rs (`arrow-arith`, `arrow-buffer`).
- AArch64 / NEON path is a planned follow-up PR — opened separately so this PR stays small and reviewable.

## Why no `check_hash` / `insert_hash` public API change

These methods exist but are `pub(crate)`. Making them `pub` would unlock significant additional perf for callers who batch-hash columns (a common pattern in Polars, DataFusion, Velox-style engines). That's a deliberately separate proposal — this PR is just the SIMD probe.

## Safety

The `unsafe` blocks are scoped to the AVX2 intrinsic calls themselves. The dispatch wrapper is safe Rust: it checks `is_x86_feature_detected!("avx2")` at runtime and only calls the AVX2 path when the feature is present. No buffer-level `unsafe` outside the intrinsic interface.

## NEON path

Deliberately not in this PR. Planned for a follow-up once this lands and the dispatch pattern is in place. AArch64 hosts get the existing scalar fallback in the meantime — no regression.

# Are there any user-facing changes?

No public API changes, no wire-format changes, no MSRV change, no new dependencies. The PR is a pure performance improvement for x86 hosts; AArch64 hosts unchanged.
