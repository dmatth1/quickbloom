#!/usr/bin/env python3
"""Benchmark the three Bloom filter implementations.

Reports min/median/p90 ns per op at multiple filter sizes spanning the
cache hierarchy (in-L2 through out-of-L3). For each candidate, both the
hash+bloom path (what a real user pays) and the prehash path (algorithm
only, comparable to published cyc/op figures) are measured.

Usage:
    python3 bench_all.py                 # all candidates, S+M+L+XL sizes
    python3 bench_all.py --sizes S,XL    # subset (endpoints only)
    python3 bench_all.py --no-prehash    # skip the prehash bench
    CC=clang python3 bench_all.py        # use clang (typically ~12% faster)
"""
import argparse
import time
from pathlib import Path

import harness


CANDIDATES = [
    ("bloom_single_key.c", "single_key  -- best in-cache, simplest"),
    ("bloom_unified.c",    "unified     -- prefetch added; good default"),
    ("bloom_batched.c",    "batched     -- best out-of-cache + 8-way API"),
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


def report_size(srcs, size, do_prehash):
    sz = harness.BENCH_SIZES[size]
    nbytes_kb = sz["nbits"] / 8 / 1024
    nbytes_str = f"{nbytes_kb:.0f} KB" if nbytes_kb < 1024 else f"{nbytes_kb/1024:.0f} MB"
    print(f"## Size {size}: {sz['nbits']:>13,} bits ({nbytes_str} filter)  "
          f"n_insert={sz['n_insert']:,}  n_query={sz['n_query']:,}")
    print()
    if do_prehash:
        head = (f"{'candidate':<24s}  "
                f"{'insert':>17}  {'miss':>17}  {'hit':>17}    "
                f"{'pre.ins':>17}  {'pre.miss':>17}  {'pre.hit':>17}    fp")
    else:
        head = (f"{'candidate':<24s}  "
                f"{'insert':>17}  {'miss':>17}  {'hit':>17}    fp")
    print(head)
    print("-" * len(head))
    reps = {"S": 15, "M": 9, "L": 5, "XL": 3}[size]
    warm = {"S": 3, "M": 2, "L": 2, "XL": 2}[size]
    for src, _label in srcs:
        if not Path(src).exists():
            print(f"{src:<24s}  (not found)")
            continue
        try:
            c = harness.compile_candidate(Path(src), Path("runs/bench"))
            d = harness.diff_test(c, nbits=sz["nbits"], n_insert=sz["n_insert"],
                                  n_query=min(sz["n_query"], 200_000))
        except Exception as e:
            print(f"{src:<24s}  ERROR: {e}")
            continue
        if not d.passed:
            print(f"{src:<24s}  FAIL: {d.reason}")
            continue
        b = harness.benchmark_at(c, size, repeats=reps, warmup=warm)
        row = (f"{Path(src).stem:<24s}  "
               f"{fmt(b.insert):>17}  {fmt(b.miss):>17}  {fmt(b.hit):>17}")
        if do_prehash:
            bp = harness.benchmark_prehash_at(c, size, repeats=reps, warmup=warm)
            if bp:
                row += f"    {fmt(bp.insert):>17}  {fmt(bp.miss):>17}  {fmt(bp.hit):>17}"
            else:
                row += "    " + ("(no prehash)").rjust(58)
        row += f"    {d.fp_rate:.5f}"
        print(row)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="S,M,L,XL",
                    help="comma-separated subset of S,M,L,XL "
                         "(default: full sweep S,M,L,XL across cache regimes)")
    ap.add_argument("--no-prehash", action="store_true")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    for s in sizes:
        if s not in harness.BENCH_SIZES:
            raise SystemExit(f"unknown size {s!r}; valid: {','.join(harness.BENCH_SIZES)}")

    header()
    t0 = time.perf_counter()
    for size in sizes:
        report_size(CANDIDATES, size, do_prehash=not args.no_prehash)
    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
