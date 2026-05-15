# Upstreaming v14 to the Parquet ecosystem: PR strategy

Companion to `PARQUET.md` and `parquet_port/`. While those document the
technical finding (the entire native Parquet ecosystem ships scalar
SBBF probes) and ship the working AVX2 drop-in with diff-test guarantee,
this document captures the strategic question: **is it worth opening
the PRs?**

Short answer: yes, but the case is sharper than "free CPU at scale,"
and the honest cost is bigger than "we already wrote the code."

## Scaling the 224 µs/query honestly

`parquet_port/RESULTS.md` measured ~224 µs saved per `SELECT WHERE col = 'X'`
query against a 1 GiB-filter-footprint table (16k row groups). At face
value:

| Query volume | CPU saved/day | What it actually is |
|---|---|---|
| 100k queries/day (small org) | 22 sec | invisible |
| 10M queries/day (mid analytics co) | 37 min | ~0.25 core continuously |
| 100M queries/day (Snowflake/BigQuery-class) | ~6 hours | ~16 cores, ~$10k/month at cloud prices |
| 1B queries/day (Cloudflare-class log analytics) | ~62 hours | ~160 cores, ~$100k/month |

Two big caveats to keep this honest:

1. **224 µs is the *large* regime.** Most production queries don't
   probe 16k row groups per value — they probe maybe 10–100. The
   medium regime saves 53 µs/query and even that's contingent on the
   query touching many row groups.
2. **Bloom is 2–5% of total query CPU** in most realistic engines.
   I/O, decompression, scan, project all dominate. A 3× bloom
   speedup is ~1–3% on total wall time. Users don't notice individual
   queries.

So the framing "real CPU time at scale" is true but smaller than the
headline suggests. The case for doing the PRs anyway: **open source
compounds.** A patch in Apache Arrow C++ ships to ~20 downstream
engines and runs for 5+ years. The total CPU-years saved is the
integral, not the slice.

## Why these PRs are unusually low-cost

Most performance PRs face a hard review tax because reviewers must
convince themselves the change is correct under all conditions. v14
sidesteps this:

1. **The implementation is already written and diff-tested.** 0
   mismatches across 167M (query, filter) pairs in `parquet_port/`.
   The bit-identical guarantee makes review trivial — the reviewer
   verifies that `vpmulld + vpsrld + vpsllvd` produces the same 8 bit
   positions as the scalar `(key * SALT[i]) >> 27` loop. 5-minute
   math check.
2. **Behind a CPU feature check, AVX2 is opt-in.** Non-AVX2 builds
   (ARM, RISC-V, old x86) keep the scalar path. No portability
   regression.
3. **The benchmark is in the patch.** Reviewers don't have to
   construct their own — `parquet_port/` is exactly what they'd ask
   for.
4. **No format change.** Cross-engine interop is preserved. A filter
   written by Spark is still readable by DuckDB; only the *probe path*
   changes.

## The realistic ROI per project

| Target | Land time | Probability | Downstream blast |
|---|---|---|---|
| `apache/arrow-rs` (Rust) | 1–3 months | 70–80% | Polars + DataFusion + Rust analytics |
| `apache/arrow` (C++) | 3–6 months | 60–70% | DuckDB + ClickHouse + pyarrow → pandas + StarRocks + Doris |
| `facebookincubator/velox` | 2–4 months | 50–60% | Trino + Presto + Spark-native via Prestissimo |

Probabilities are estimates, not gospel. arrow-rs is fastest because
the project is smaller and the Rust community has lower friction.
Apache Arrow C++ carries Apache process overhead (ICLA, dev@ mailing
list, formal review). Velox is Meta-led and timelines depend on team
priorities.

## The hidden costs people forget

Things that make "it's open source, just open a PR" undersell the
real cost:

1. **Time cost.** Each PR is roughly 1–2 weeks initial work plus 1–3
   months of intermittent review. Not free.
2. **CLA / employer IP review.** If you contribute as part of a job,
   your employer may need to sign off on assigning copyright to
   Apache (ICLA for individuals or CCLA for corporations). Real
   friction, 1–2 weeks first time.
3. **Maintenance tail.** Accepted code creates a small ongoing
   obligation — bug reports, regressions in CI, ports to new
   architectures.
4. **Review iteration burnout.** Apache projects sometimes ask for
   5–10 review rounds. Some land in 2 weeks; some take a year.
5. **Reviewers might prefer alternatives.** "We're moving to format
   v2" or "we want to land Ribbon first" can sideline a PR
   indefinitely.

These don't kill the case — they shape the realistic expectation: "9
months from now, three patches landed, ~3× speedup shipped to ~20
engines, contributor has Apache Arrow + arrow-rs + Velox on their
resume." Great outcome, but it's a quarter of a year, not a weekend.

## arrow-rs vs Apache Arrow C++: the language question

A reasonable instinct is to go straight to Apache Arrow C++ because
the existing `parquet_port/` code is in C++. That's correct for the
code part, but wrong for the process part.

### The Rust "rewrite" is mostly mechanical

The AVX2 intrinsics in Rust use the **identical names** as C++ via
`std::arch::x86_64`:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_avx2(block: &[u32; 8], hash: u32) -> bool {
    use std::arch::x86_64::*;
    let hash_v = _mm256_set1_epi32(hash as i32);
    let salt = _mm256_loadu_si256(SALT.as_ptr() as *const __m256i);
    let prod = _mm256_mullo_epi32(hash_v, salt);
    let shift = _mm256_srli_epi32(prod, 27);
    let ones = _mm256_set1_epi32(1);
    let mask = _mm256_sllv_epi32(ones, shift);
    let blk = _mm256_loadu_si256(block.as_ptr() as *const __m256i);
    _mm256_testc_si256(blk, mask) != 0
}
```

~30 lines including runtime dispatch. Direct line-by-line translation
from `parquet_port/src/parquet_bloom.cpp`. Half a day, not a week.
Mostly you'd be learning cargo bench harness and clippy conventions,
not learning Rust.

### Apache Arrow C++ patch surface

Code transfer is the easy part. What's actually involved:

1. **New file** `cpp/src/parquet/bloom_filter_avx2.cc`. Apache Arrow
   conventions: SIMD variants live in separate translation units.
   Precedent: `arrow/util/bit_util_avx2.cc`, `arrow/util/hashing_avx2.cc`.
2. **CpuInfo runtime dispatch.** Apache Arrow supports ARM
   (Graviton), RISC-V, old x86 — they don't compile-time gate on
   AVX2. Dispatch via `CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2)`,
   integrated with their `DispatchLevel` function-pointer pattern.
3. **Tests** in `parquet-bloom-filter-test`. They want the AVX2 path
   tested explicitly even though it's bit-identical — Apache convention.
4. **Bench** in `parquet-bloom-filter-benchmark` so reviewers see
   the gain in their own CI.
5. **`dev@arrow.apache.org` discussion** before the PR for non-trivial
   changes. They prefer pre-discussion for SIMD additions. Realistically
   a 2–3 message thread to confirm the maintainers want this.
6. **ICLA.** First-time contributors need this on file.

So the patch is maybe 200 lines of new code (vs the ~30 lines of
"actual" AVX2), with the rest being plumbing into their dispatch /
test / bench infrastructure.

### Realistic timeline for Apache Arrow C++

| Stage | Time |
|---|---|
| ICLA (if needed) | 1–2 weeks |
| `dev@` discussion | 1 week |
| Write the patch + tests + bench | 3–5 days |
| Initial review | 2–4 weeks |
| Review iterations (typically 3–6 rounds) | 6–12 weeks |
| **Total to merge** | **3–5 months** |

The 3–5 days of actual coding is dwarfed by the process. That's not
Apache being slow — it's a mature project with high review standards.

### The trade-off

| Axis | Apache Arrow C++ | arrow-rs |
|---|---|---|
| Code translation | None — direct from `parquet_port/` | ~30 lines mechanical |
| Project process | Heavier (ICLA, dev@, multi-round review) | Lighter |
| Realistic time-to-merge | 3–5 months | 1.5–2.5 months |
| Blast radius | DuckDB, ClickHouse, pyarrow, pandas, StarRocks, Doris | Polars, DataFusion, Rust ecosystem |
| Reputation value | High (Apache Arrow proper) | High (Polars / DataFusion visibility) |

If the goal is **biggest immediate downstream**: Apache Arrow C++.
If the goal is **shortest path to landed code**: arrow-rs.

## Recommended sequence

For a contributor who wants Apache Arrow C++ as the first target
(direct code reuse, biggest blast radius), the recommended sequence:

1. **Send a `dev@arrow.apache.org` email first.** Link `PARQUET.md`
   + `parquet_port/RESULTS.md`. Tone: "I have a working, diff-tested
   AVX2 implementation of `FindHash` with 3× in-cache and
   1.15–1.5× out-of-L3 speedup. Would you accept this?" Don't open
   the PR yet. Costs nothing, signal in 1–2 weeks.
2. **If they say yes**: write the patch, deal with ICLA, expect
   3–5 months to merge.
3. **If they say "maybe, but we want X"**: respond, refine, ship.
4. **If they say no** (unlikely given bit-identical): blog post +
   drop-in patch repo.

The `dev@` first move is important because it surfaces hidden
objections (e.g., "we're waiting for Parquet v2 to deprecate SBBF"
— they're not, but if they were, you'd want to know before writing
the patch).

### Then the arrow-rs follow-up

Once Apache Arrow C++ lands, an arrow-rs PR becomes trivial because
it has a precedent to point at: "Apache Arrow C++ shipped this in
version X.Y, here's the Rust equivalent." Even maintainers who'd
push back on a standalone PR usually wave through "match upstream
Arrow." Timeline: ~1 month instead of ~3.

So the recommended order: **Apache Arrow C++ dev@ → patch → merge →
arrow-rs follow-up referencing the C++ patch → optional Velox**.

## One caveat: AVX-512

Apache Arrow C++ reviewers may ask for **AVX-512** too since they
support it (`ARROW_HAVE_AVX512`). v14 deliberately scoped to AVX2 for
AMD compatibility (drops Zen 3 and older AMD if AVX-512 becomes the
only path).

Quick rebuttal ready: same algorithm, wider registers, marginal
additional speedup on Skylake-X+/Zen 4+, but you'd need it behind
another dispatch level so AMD pre-Zen 4 keeps the AVX2 path. Doable
as a second commit if they push, but the AVX2 patch is the right
first step because it covers the bigger hardware envelope.

## Alternative: don't do PRs, do a blog post

A blog post with the bench results, the diff-test guarantee, and
source-quoted screenshots of arrow-cpp / arrow-rs / Velox shipping
scalar code is sometimes *more* effective at getting the patch landed
than opening the PR yourself. Project maintainers see the analysis,
realize the win is easy, and ship it themselves. Less work, faster
outcome, less name-on-the-PR but more total impact.

The trade-off: you don't get the resume credit. The maintainers ship
it under their own attribution.

If the goal is **make the ecosystem faster as fast as possible**:
blog post route delivers ~80% of the impact for ~20% of the effort.

If the goal is **portfolio credit + landed contribution**: open the
PRs.

## TL;DR

Worth pursuing? Yes — the case compounds across 20+ downstream
engines and 5+ years of usage. But "open source, why not" undersells
the real time cost (3–5 months for Apache Arrow C++, 1.5–2.5 months
for arrow-rs). The diff-test bit-identical guarantee makes these
PRs unusually low-risk; the process overhead is what costs time.

Recommended start: `dev@arrow.apache.org` email referencing
`PARQUET.md` and `parquet_port/RESULTS.md`. Then write the C++ patch
on confirmed signal. arrow-rs follows naturally as a second PR
referencing the first.
