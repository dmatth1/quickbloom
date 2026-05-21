#!/usr/bin/env python3
"""Benchmark the Bloom filter implementations.

Reports min/median/p90 ns per op at multiple filter sizes spanning the
cache hierarchy (in-L2 through out-of-L3). For each candidate, both the
hash+bloom path (what a real user pays) and the prehash path (algorithm
only, comparable to published cyc/op figures) are measured.

Two sizing modes:
  * Fixed-size (default): every candidate gets the same nbits at each
    preset. Convenient, but FP rates differ across candidates because
    they use different K. Use this to compare raw kernel cost.
  * Equal-FP (--target-fp 0.001): each candidate is sized so its
    theoretical FP rate hits the target given n_insert and its K. This
    is the apples-to-apples comparison for "what does it cost to hit
    FP = X".

Usage:
    python3 bench_all.py                       # the 3 top-level designs
    python3 bench_all.py --comparisons         # also include comparisons/*.c
    python3 bench_all.py --sizes S,XL          # subset of sizes
    python3 bench_all.py --no-prehash          # skip the prehash bench
    python3 bench_all.py --target-fp 0.001     # equal-FP mode
    CC=clang python3 bench_all.py              # use clang (typically ~12% faster)
"""
import argparse
import math
import time
from pathlib import Path

import harness


# The 3 designs this repo ships at the top level.
MAIN_CANDIDATES = [
    ("bloom_single_key.c", "single_key  -- SBBF, no prefetch (best in-cache)"),
    ("bloom_unified.c",    "unified     -- SBBF + prefetch (good default)"),
    ("bloom_batched.c",    "batched     -- 64-bit blocks + 8-way SIMD"),
]

# External references / baselines, lifted from comparisons/ when --comparisons.
COMPARISON_CANDIDATES = [
    ("comparisons/bloom_impala.c",
     "impala      -- Apache Parquet/Impala SBBF, scalar + XXH64"),
    ("comparisons/bloom_krassovsky.c",
     "krassovsky  -- Save Buffer PatternedSimd, 64-bit + gather"),
    ("comparisons/bloom_classic.c",
     "classic     -- textbook K-hash bloom (baseline)"),
    ("comparisons/bloom_xorfuse.c",
     "xorfuse     -- Graf+Lemire binary fuse filter (static; bloom-ABI shim)"),
    ("comparisons/fastbloom_shim",
     "fastbloom   -- tomtomwombat/fastbloom (Rust; SipHash-1-3 + concurrency)"),
    ("comparisons/arrow_rs_sbbf_shim",
     "arrow_rs    -- arrow-rs parquet::bloom_filter::Sbbf (Rust; XXH64; no prehash API)"),
    ("comparisons/quickbloom_arrow_rs_shim",
     "qb_rs       -- quickbloom Rust port: AVX2 vptest + XXH64 (head-to-head vs arrow_rs)"),
]


def fmt(stat):
    return f"{stat.min:5.2f}/{stat.median:5.2f}/{stat.p90:5.2f}"


def header():
    info = harness.cpu_info()
    print("# Bloom filter benchmark")
    print(f"# CPU:    {info['model']}  @ {info['mhz']:.0f} MHz")
    print(f"# Cache:  L1d {info['L1d_kb']} KB / L2 {info['L2_kb']} KB / L3 {info['L3_mb']} MB")
    print(f"# Build:  {harness.CC} -O3 -mavx2 -mbmi2 -mfma -maes")
    print(f"# Stats:  min/median/p90 ns/op (lower is better)")
    print()


def nbits_for_target_fp(target_fp: float, n_insert: int, k: int) -> int:
    """Smallest pow-of-2 nbits such that theoretical FP(k, n, m) <= target_fp.

    FP = (1 - exp(-k*n/m))^k  =>  m = -k*n / ln(1 - target^(1/k))
    """
    t = target_fp ** (1.0 / k)
    if t >= 1.0:
        raise ValueError("target_fp too large for chosen k")
    m_real = -k * n_insert / math.log(1.0 - t)
    pow2 = 1
    while pow2 < m_real:
        pow2 <<= 1
    return pow2


def size_for_candidate(c, size_key, target_fp):
    """Return (nbits, n_insert, n_query, bytes_label) for a candidate.

    If target_fp is None: fixed-size from BENCH_SIZES.
    Else: scale nbits so candidate hits target_fp at the preset's n_insert.
    """
    sz = harness.BENCH_SIZES[size_key]
    n_insert = sz["n_insert"]
    n_query = sz["n_query"]
    if target_fp is None:
        nbits = sz["nbits"]
    else:
        nbits = nbits_for_target_fp(target_fp, n_insert, c.k_hashes)
    nbytes_kb = nbits / 8 / 1024
    if nbytes_kb < 1024:
        label = f"{nbytes_kb:.0f} KB"
    else:
        label = f"{nbytes_kb/1024:.1f} MB"
    return nbits, n_insert, n_query, label


def bench_one(c, nbits, n_insert, n_query, reps, warm, do_prehash):
    """Run benchmark + diff_test for a single (candidate, sizing) pair."""
    d = harness.diff_test(c, nbits=nbits, n_insert=n_insert,
                          n_query=min(n_query, 200_000))
    b = harness.benchmark(c, nbits=nbits, n_insert=n_insert,
                          n_query=n_query, repeats=reps, warmup=warm)
    bp = None
    if do_prehash:
        bp = harness.benchmark_prehash(c, nbits=nbits, n_insert=n_insert,
                                       n_query=n_query, repeats=reps, warmup=warm)
    return d, b, bp


def report_size(srcs, size_key, do_prehash, target_fp):
    sz = harness.BENCH_SIZES[size_key]
    print(f"## Size {size_key}: n_insert={sz['n_insert']:,}  n_query={sz['n_query']:,}", end="")
    if target_fp is not None:
        print(f"  (target FP <= {target_fp})")
    else:
        nbytes_kb = sz["nbits"] / 8 / 1024
        nbytes_str = f"{nbytes_kb:.0f} KB" if nbytes_kb < 1024 else f"{nbytes_kb/1024:.0f} MB"
        print(f"  nbits={sz['nbits']:,} ({nbytes_str}, fixed)")
    print()

    if do_prehash:
        head = (f"{'candidate':<18}  {'K':>2}  {'bits/key':>8}  "
                f"{'size':>8}  "
                f"{'insert':>17}  {'miss':>17}  {'hit':>17}    "
                f"{'pre.ins':>17}  {'pre.miss':>17}  {'pre.hit':>17}    "
                f"{'fp':>7}  {'theo':>7}")
    else:
        head = (f"{'candidate':<18}  {'K':>2}  {'bits/key':>8}  "
                f"{'size':>8}  "
                f"{'insert':>17}  {'miss':>17}  {'hit':>17}    "
                f"{'fp':>7}  {'theo':>7}")
    print(head)
    print("-" * len(head))

    reps = {"S": 15, "M": 9, "L": 5, "XL": 3}[size_key]
    warm = {"S": 3, "M": 2, "L": 2, "XL": 2}[size_key]

    for src, _label in srcs:
        if not Path(src).exists():
            print(f"{src:<18}  (not found)")
            continue
        try:
            c = harness.compile_candidate(Path(src), Path("runs/bench"))
            nbits, n_insert, n_query, sz_label = size_for_candidate(c, size_key, target_fp)
            d, b, bp = bench_one(c, nbits, n_insert, n_query, reps, warm, do_prehash)
        except Exception as e:
            print(f"{src:<18}  ERROR: {e}")
            continue
        if not d.passed:
            print(f"{src:<18}  FAIL: {d.reason}")
            continue
        bits_per_key = nbits / n_insert
        row = (f"{Path(src).stem.replace('bloom_',''):<18}  "
               f"{c.k_hashes:>2}  {bits_per_key:>8.1f}  {sz_label:>8}  "
               f"{fmt(b.insert):>17}  {fmt(b.miss):>17}  {fmt(b.hit):>17}")
        if do_prehash:
            if bp:
                row += f"    {fmt(bp.insert):>17}  {fmt(bp.miss):>17}  {fmt(bp.hit):>17}"
            else:
                row += "    " + ("(no prehash)").rjust(58)
        row += f"    {d.fp_rate:>7.5f}  {d.theoretical_fp_rate:>7.5f}"
        print(row)
    print()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--sizes", default="S,M,L,XL",
                    help="comma-separated subset of S,M,L,XL "
                         "(default: full sweep)")
    ap.add_argument("--no-prehash", action="store_true",
                    help="skip the prehash bench")
    ap.add_argument("--target-fp", type=float, default=None,
                    help="equal-FP mode: size each candidate's nbits so its "
                         "theoretical FP rate hits this target given n_insert. "
                         "Apples-to-apples comparison across different K. "
                         "Example: --target-fp 0.001")
    ap.add_argument("--comparisons", action="store_true",
                    help="also benchmark comparisons/*.c (impala, krassovsky, classic)")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    for s in sizes:
        if s not in harness.BENCH_SIZES:
            raise SystemExit(f"unknown size {s!r}; valid: {','.join(harness.BENCH_SIZES)}")

    candidates = list(MAIN_CANDIDATES)
    if args.comparisons:
        candidates += COMPARISON_CANDIDATES

    header()
    if args.target_fp is not None:
        print(f"# Mode: equal-FP, target = {args.target_fp}")
        print("# Each candidate's filter is sized to hit the target FP given its K.")
        print()
    t0 = time.perf_counter()
    for size in sizes:
        report_size(candidates, size, do_prehash=not args.no_prehash,
                    target_fp=args.target_fp)
    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
