# Next steps for the arrow-rs SBBF SIMD PR

Concrete operating plan to take this branch from reference-implementation to
landed PR. Estimated 1–2 days of focused work to convert.

## 0. Pre-work (already done on this branch)

- ✅ Reference Rust port of the SIMD probe (`comparisons/quickbloom_arrow_rs_shim/`)
- ✅ Same-host bench: 2.78–3.27× on miss-heavy across S/M/L
- ✅ Decomposition: 1:1 algorithm match vs C reference; difference is hash impl
- ✅ Issue draft (`ISSUE_DRAFT.md`)
- ✅ PR description draft (`PR_DRAFT.md`)
- ✅ Honest caveats documented (faithful port not real binary; XXH64 impl-bound; etc.)

## 1. File the proposal issue (~30 min)

Optional but recommended. Lets the maintainers (Andrew Lamb, Raphael
Taylor-Davies, Daniël Heres, Will Jones, others) signal direction before
the PR shows up. Paste from `ISSUE_DRAFT.md`. Wait a day or two for at
least one ack before opening the PR.

URL: https://github.com/apache/arrow-rs/issues/new/choose

Alternative: skip the issue and go straight to the PR. arrow-rs's culture
accepts well-scoped perf PRs cold; the issue is etiquette, not blocker.

## 2. Fork and branch

```sh
gh repo fork apache/arrow-rs
cd arrow-rs
git checkout -b parquet-bloom-simd
```

## 3. Port the reference implementation (~4 hours)

Touch `parquet/src/bloom_filter/mod.rs` only. The structural diff:

### a) Add AVX2 helpers under a cfg-guard

```rust
#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use std::arch::x86_64::*;
    use super::Block;

    const SALT: [u32; 8] = [
        0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d,
        0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
    ];

    #[target_feature(enable = "avx2")]
    pub unsafe fn mask(hash: u32) -> __m256i {
        let salt = _mm256_loadu_si256(SALT.as_ptr() as *const __m256i);
        let h = _mm256_set1_epi32(hash as i32);
        let prod = _mm256_mullo_epi32(h, salt);
        let shift = _mm256_srli_epi32::<27>(prod);
        let ones = _mm256_set1_epi32(1);
        _mm256_sllv_epi32(ones, shift)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn block_check(block: &Block, hash: u32) -> bool {
        let m = mask(hash);
        let b = _mm256_loadu_si256(block.as_ptr() as *const __m256i);
        _mm256_testc_si256(b, m) != 0
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn block_insert(block: &mut Block, hash: u32) {
        let m = mask(hash);
        let p = block.as_mut_ptr() as *mut __m256i;
        let b = _mm256_loadu_si256(p);
        _mm256_storeu_si256(p, _mm256_or_si256(b, m));
    }
}
```

### b) Rename existing scalar functions to `_scalar` suffix

The current `Block::check` / `Block::insert` become `block_check_scalar`
/ `block_insert_scalar` (still public to the module, fallback for non-AVX2).

### c) Dispatch wrappers

```rust
fn block_check(block: &Block, hash: u32) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { simd_x86::block_check(block, hash) };
        }
    }
    block_check_scalar(block, hash)
}

fn block_insert(block: &mut Block, hash: u32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { simd_x86::block_insert(block, hash) };
        }
    }
    block_insert_scalar(block, hash);
}
```

### d) Wire the existing public `check_hash` / `insert_hash` to use the dispatch wrappers

(They currently inline the scalar code; change them to call `block_check` / `block_insert`.)

## 4. Add tests (~1 hour)

Inside the existing `#[cfg(test)] mod tests` in `bloom_filter/mod.rs`:

```rust
#[test]
#[cfg(target_arch = "x86_64")]
fn simd_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") { return; }
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    for _ in 0..10_000 {
        let mut block: Block = [0u32; 8];
        for w in &mut block {
            *w = rng.gen();
        }
        let hash: u32 = rng.gen();
        let s = block_check_scalar(&block, hash);
        let v = unsafe { simd_x86::block_check(&block, hash) };
        assert_eq!(s, v, "scalar vs SIMD differ for hash {:x}", hash);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn simd_insert_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") { return; }
    let mut rng = StdRng::seed_from_u64(0xBABE);
    for _ in 0..10_000 {
        let hash: u32 = rng.gen();
        let mut block_scalar: Block = [0u32; 8];
        let mut block_simd: Block = [0u32; 8];
        block_insert_scalar(&mut block_scalar, hash);
        unsafe { simd_x86::block_insert(&mut block_simd, hash); }
        assert_eq!(block_scalar, block_simd);
    }
}
```

## 5. Add a benchmark (~1 hour)

Currently no `parquet/benches/bloom_filter_check.rs` exists. Add one based
on the `arrow_writer.rs` bench style. Three sizes (S/M/L matching our
test harness), miss-heavy and hit-heavy workloads, `Sbbf::check` only
(the public API). Bench numbers in the PR description should come from
running this on the contributor's hardware, not from our reference
branch's harness.

## 6. Local validation

```sh
cd arrow-rs
cargo test -p parquet -- bloom_filter
cargo clippy -p parquet --all-features
cargo bench -p parquet --bench bloom_filter_check
```

All must pass / show the expected speedup.

## 7. Open the PR

```sh
git push -u origin parquet-bloom-simd
gh pr create --title "parquet: SIMD-accelerate Sbbf probe (AVX2; 2.4–3.3× speedup)" \
  --body "$(cat path/to/PR_DRAFT.md)"
```

(Replace placeholder issue number in `PR_DRAFT.md` with the actual issue
number from step 1.)

## 8. Iterate on review

Likely review feedback:

- Maintainer might prefer `wide` over `std::arch` — be ready with the
  perf-vs-deps tradeoff argument.
- They might ask for NEON in the same PR — push back gently with
  "happy to follow up, but this PR is scoped to x86 to keep review
  surface small."
- They might ask for additional benchmarks on Apple Silicon / Graviton
  — note that the scalar path is unchanged there.
- They might want `check_hash`/`insert_hash` made `pub` in the same PR
  — same answer: separate follow-up.

## Honest time estimate

- Step 1 (issue): 30 min
- Steps 2–5 (port + tests + bench): 4–6 hours
- Step 6 (local validation): 30 min
- Step 7 (open PR): 15 min
- Step 8 (review cycles): 1–4 weeks of wall time, low per-cycle effort

Total active work: **~1 day**. Wall time to merged: **2–6 weeks**
depending on review-cycle latency.
