# arrow-rs SIMD bloom filter PR — handoff

Single-doc handoff for the session that will fork `apache/arrow-rs`,
port our reference implementation into their codebase, and open the PR.
Self-contained: everything you need is in this document. Other files in
this directory (`PR_DRAFT.md`, `ISSUE_DRAFT.md`, `NEXT_STEPS.md`,
`README.md`) are supporting references.

## Goal

Open a single PR to `apache/arrow-rs` adding an AVX2 SIMD path to the
Parquet bloom filter probe, with the existing scalar code preserved as
a runtime fallback. **Decision: skip the discussion thread, file the PR
directly with strong bench evidence in the description.** This matches
arrow-rs's PR culture (see PR #1221, #1248 for precedent on SIMD perf
work landing cold).

## What's been validated already

Reference Rust port lives at `../../comparisons/quickbloom_arrow_rs_shim/`
in this repo, built as a `cdylib` exposing the same C ABI as the existing
`arrow_rs_sbbf_shim`. Same hash, same algorithm, same wire format — only
the inner SIMD operations differ.

**Same-host A/B** (Intel Xeon @ 2.1 GHz Sapphire Rapids, contains-miss
hash+bloom path, median ns/op):

| Regime | Filter size | arrow-rs `Sbbf` (scalar) | Reference SIMD port | Speedup |
|--------|------------:|-------------------------:|--------------------:|--------:|
| S | 128 KB (in L2) | 18.52 | **5.66** | **3.27×** |
| M | 2 MB (in L3) | 23.75 | **8.55** | **2.78×** |
| L | 32 MB (around L3) | 30.59 | **12.53** | **2.44×** |

Insert path: 1.06× / 1.16× / 1.17× across S/M/L. Hit-heavy is roughly
tied because `XxHash64::oneshot` dominates the call time on hits; the
SIMD probe win is fully visible on the miss-heavy workload that
dominates real Parquet row-group skipping.

Raw captures: `results/run5_post_hash_fix.txt` (S+M) and
`results/run6_lxl_post_fix.txt` (L).

## What you need to do — step by step

### Step 1: fork and clone (~5 min)

```sh
gh repo fork apache/arrow-rs --clone
cd arrow-rs
git checkout -b parquet-bloom-simd
```

### Step 2: port the SIMD path into `parquet/src/bloom_filter/mod.rs` (~3 hours)

Open the file. The current scalar `Block::check` / `Block::insert`
look like this:

```rust
impl Block {
    fn check(&self, hash: u32) -> bool {
        let mask = Self::mask(hash);
        for i in 0..8 {
            if self[i] & mask[i] == 0 { return false; }
        }
        true
    }

    fn insert(&mut self, hash: u32) {
        let mask = Self::mask(hash);
        for i in 0..8 { self[i] |= mask[i]; }
    }
}
```

Apply these changes:

**(a) Rename the scalar versions** to `check_scalar` / `insert_scalar`
(keep them — they're the fallback for non-x86 / non-AVX2 hosts).

**(b) Add a new SIMD module** at the top of the file, gated on `cfg(target_arch = "x86_64")`:

```rust
#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use std::arch::x86_64::*;
    use super::Block;

    // Same Parquet SBBF SALT as the scalar `Block::SALT` constant.
    // Loaded as a vector for the K=8 mask compute.
    const SALT: [u32; 8] = [
        0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d,
        0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
    ];

    /// AVX2 K=8 mask vector. Lowers to: vmovdqu + vpmulld + vpsrld + vpsllvd.
    #[target_feature(enable = "avx2")]
    unsafe fn mask(hash: u32) -> __m256i {
        let salt = _mm256_loadu_si256(SALT.as_ptr() as *const __m256i);
        let h = _mm256_set1_epi32(hash as i32);
        let prod = _mm256_mullo_epi32(h, salt);
        let shifts = _mm256_srli_epi32::<27>(prod);
        let ones = _mm256_set1_epi32(1);
        _mm256_sllv_epi32(ones, shifts)
    }

    /// SIMD per-block check using `_mm256_testc_si256` (vptest). Same
    /// semantics as scalar: returns true iff every mask bit is set in
    /// the block.
    #[target_feature(enable = "avx2")]
    pub unsafe fn check(block: &Block, hash: u32) -> bool {
        let m = mask(hash);
        let blk = _mm256_loadu_si256(block.as_ptr() as *const __m256i);
        _mm256_testc_si256(blk, m) != 0
    }

    /// SIMD per-block insert: vector load + or + store.
    #[target_feature(enable = "avx2")]
    pub unsafe fn insert(block: &mut Block, hash: u32) {
        let m = mask(hash);
        let p = block.as_mut_ptr() as *mut __m256i;
        let blk = _mm256_loadu_si256(p);
        _mm256_storeu_si256(p, _mm256_or_si256(blk, m));
    }
}
```

**(c) Replace `Block::check` / `Block::insert` body** with the
dispatch wrapper:

```rust
impl Block {
    #[inline]
    fn check(&self, hash: u32) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return unsafe { simd_x86::check(self, hash) };
            }
        }
        self.check_scalar(hash)
    }

    #[inline]
    fn insert(&mut self, hash: u32) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return unsafe { simd_x86::insert(self, hash) };
            }
        }
        self.insert_scalar(hash);
    }

    fn check_scalar(&self, hash: u32) -> bool {
        let mask = Self::mask(hash);
        for i in 0..8 {
            if self[i] & mask[i] == 0 { return false; }
        }
        true
    }

    fn insert_scalar(&mut self, hash: u32) {
        let mask = Self::mask(hash);
        for i in 0..8 { self[i] |= mask[i]; }
    }
}
```

Reference: `../../comparisons/quickbloom_arrow_rs_shim/src/lib.rs` has
the same algorithm in a cdylib form. The arrow-rs port differs only in:
- It's inside the `Block` impl rather than a top-level function
- It uses arrow-rs's `Block` type alias rather than the standalone
  `AlignedU32Buf` we built (their Block is already 256-bit-aligned via
  `#[repr(align(32))]`)
- It doesn't carry the C ABI shim — just the `impl Block` methods

### Step 3: add tests (~1 hour)

Add to the existing `#[cfg(test)] mod tests` in `bloom_filter/mod.rs`:

```rust
#[cfg(test)]
mod simd_tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// Verify the AVX2 path produces bit-identical output to the
    /// scalar path across 10K random (block, hash) pairs.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_check_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return; // bench host without AVX2 — skip
        }
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        for _ in 0..10_000 {
            let block: Block = [
                rng.r#gen(), rng.r#gen(), rng.r#gen(), rng.r#gen(),
                rng.r#gen(), rng.r#gen(), rng.r#gen(), rng.r#gen(),
            ];
            let hash: u32 = rng.r#gen();
            let s = block.check_scalar(hash);
            let v = unsafe { simd_x86::check(&block, hash) };
            assert_eq!(s, v, "scalar={s} simd={v} for hash={hash:#x}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_insert_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(0xBABE);
        for _ in 0..10_000 {
            let hash: u32 = rng.r#gen();
            let mut b_scalar: Block = [0; 8];
            let mut b_simd: Block = [0; 8];
            b_scalar.insert_scalar(hash);
            unsafe { simd_x86::insert(&mut b_simd, hash); }
            assert_eq!(b_scalar, b_simd, "scalar vs SIMD insert differ for hash {hash:#x}");
        }
    }
}
```

If arrow-rs uses a different rng convention (e.g. they prefer
`SmallRng` or roll their own), match their style — grep their existing
tests for `rng` to see.

### Step 4: add a benchmark (~1 hour)

There's no existing `parquet/benches/bloom_filter_check.rs` — you'll add
it. Use `parquet/benches/arrow_writer.rs` as a style template. Three
sizes (1 KB / 1 MB / 32 MB) crossed with miss-heavy / hit-heavy.

Sketch:

```rust
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use parquet::bloom_filter::Sbbf;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn bench_check(c: &mut Criterion) {
    for &(label, nbytes, n_insert, n_query) in &[
        ("s_1KiB",    1 * 1024,           1_000,    10_000),
        ("m_1MiB",    1 * 1024 * 1024,  100_000,   100_000),
        ("l_32MiB",  32 * 1024 * 1024,  500_000,   500_000),
    ] {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);

        // Build the filter
        let mut filter = Sbbf::new_with_num_of_bytes(nbytes);
        let mut inserted: Vec<[u8; 16]> = Vec::with_capacity(n_insert);
        for _ in 0..n_insert {
            let mut k = [0u8; 16]; rng.fill(&mut k);
            inserted.push(k);
            filter.insert(&k.as_slice());
        }

        // Disjoint miss-set
        let miss_keys: Vec<[u8; 16]> = (0..n_query).map(|_| {
            let mut k = [0u8; 16]; rng.fill(&mut k); k[0] |= 0x80; k
        }).collect();
        let hit_keys: Vec<[u8; 16]> = (0..n_query).map(|_| {
            inserted[rng.gen_range(0..inserted.len())]
        }).collect();

        c.bench_function(&format!("sbbf_check_miss_{label}"), |b| {
            b.iter(|| {
                for k in &miss_keys {
                    black_box(filter.check(&k.as_slice()));
                }
            });
        });
        c.bench_function(&format!("sbbf_check_hit_{label}"), |b| {
            b.iter(|| {
                for k in &hit_keys {
                    black_box(filter.check(&k.as_slice()));
                }
            });
        });
    }
}

criterion_group!(benches, bench_check);
criterion_main!(benches);
```

Add to `parquet/Cargo.toml` under `[[bench]]`:

```toml
[[bench]]
name = "bloom_filter_check"
harness = false
```

If `criterion` isn't already a `[dev-dependencies]` for the parquet
crate, check whether arrow-rs benches use a different harness
(`hint::black_box` + `Instant::now`-based, or whatever) and match.
Run `grep -A 2 '^\[\[bench\]\]' parquet/Cargo.toml` to see precedent.

### Step 5: local validation (~30 min)

```sh
cd arrow-rs

# Tests must pass — including the new SIMD-vs-scalar diff tests
cargo test -p parquet -- bloom_filter

# Clippy must be clean for the new code
cargo clippy -p parquet --all-features -- -D warnings

# Bench should show the 2.4-3.3x speedup
cargo bench -p parquet --bench bloom_filter_check
```

Capture the bench output — you'll paste a fresh table from the host you
run on into the PR description (don't reuse the table from this doc,
which is from a virtualised host; use whatever the contributor's
hardware reports).

### Step 6: file the PR (~15 min)

```sh
git add -A
git commit -m "parquet: SIMD-accelerate Sbbf probe (AVX2; 2.4-3.3x speedup)"
git push -u origin parquet-bloom-simd
gh pr create \
  --title "parquet: SIMD-accelerate Sbbf probe (AVX2; 2.4-3.3x speedup)" \
  --body "$(cat path/to/PR_DRAFT.md)"
```

Use `PR_DRAFT.md` in this directory as the PR body. Update the
benchmark table in it with your fresh numbers before pasting.

### Step 7: anticipate review feedback

Likely review questions and pre-prepared responses:

| Reviewer might ask | Response |
|--------------------|----------|
| "Why not use `wide` / `std::simd`?" | `std::simd` is nightly-only (MSRV). `wide` adds a dep + ~5% perf cost. `std::arch` matches the existing SIMD pattern in `arrow-arith` / `arrow-buffer`. Happy to switch if the project prefers `wide`. |
| "What about ARM / NEON?" | Separate follow-up PR. Scalar fallback runs on AArch64 unchanged — no regression. Keeping this PR scoped to x86 for review clarity. |
| "Should `check_hash` / `insert_hash` be public?" | Worth doing — would unlock batched-hash perf for callers like Polars / DataFusion that already have column hashes. Deliberately separate from this PR; happy to file a follow-up. |
| "MSRV bump?" | None needed. `std::arch::x86_64` and `is_x86_feature_detected!` stable since Rust 1.27. |
| "Test coverage?" | 20K diff-tests (10K check + 10K insert) on the new diff verifying SIMD ↔ scalar identical output. Plus all existing bloom_filter tests continue passing through the dispatch wrapper. |
| "Bench numbers on my hardware?" | The bench harness ships with the PR. Reviewers can `cargo bench -p parquet --bench bloom_filter_check` on their own hardware. |

## What could go wrong

- **`Block` type alias details**: arrow-rs's `Block` is currently `[u32; 8]`.
  If they've changed it to `[u32; 8]` wrapped in `#[repr(align(32))]`,
  the SIMD code is unchanged; if they've abstracted it differently,
  the `.as_ptr()` / `.as_mut_ptr()` calls may need adjustment.
- **`Self::mask` vs `simd_x86::mask`**: the scalar Self::mask function
  already exists and returns `[u32; 8]`. The SIMD `mask` computes the
  same values but returns `__m256i`. Keep both; they're not in conflict.
- **`rand` crate edition**: rust-2024 reserves `gen` as a keyword, so
  `rng.gen()` becomes `rng.r#gen()` in arrow-rs's current MSRV. Match
  whatever they do — `grep -rn 'rng\.' parquet/src/` for precedent.
- **Bench harness**: if criterion isn't already a dev-dep for parquet,
  there may be friction adding it. Check first; might need to use
  whatever lightweight harness they prefer.

## Honest time estimate

- Steps 1–6: **~1 day** of focused work
- Step 7 (review cycles): **2–6 weeks** wall time, low per-cycle effort

## Why this is worth doing

Single PR → cascades to: Polars, Databend, InfluxDB 3.0 / IOx, Quickwit,
LanceDB, RisingWave, GreptimeDB, every DataFusion-stack downstream. They
all benefit on the next arrow-rs minor version. No coordination, no
config flips, no version negotiation.

This is the best leverage ratio on the entire `APPLICATIONS.md` deck —
one day of focused work, cascading to 7+ named downstream projects, no
new dependencies, no MSRV bump, no political process.

## Files referenced from this doc

- `comparisons/quickbloom_arrow_rs_shim/src/lib.rs` — reference Rust impl
- `comparisons/quickbloom_arrow_rs_shim/Cargo.toml` — uses `twox-hash = "2"` with `xxhash64` feature
- `results/run5_post_hash_fix.txt` — S+M bench captures
- `results/run6_lxl_post_fix.txt` — L bench captures
- `PR_DRAFT.md` — PR body to paste at filing time
- `ISSUE_DRAFT.md` — proposal issue body (skip if going PR-first)
- `NEXT_STEPS.md` — earlier operating-plan version (superseded by this doc)
- `README.md` — investigation overview
