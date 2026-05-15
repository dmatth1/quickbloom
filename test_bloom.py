#!/usr/bin/env python3
"""Correctness tests for the three Bloom filter implementations.

Compiles each candidate and runs the differential test:
  - inserts N random 16-byte keys, then queries them: expect 0 false negatives
  - queries N unseen random keys: expect false positive rate within spec

Tests at multiple filter sizes to catch cache-regime-specific bugs.
"""
import sys
from pathlib import Path

import harness


CANDIDATES = [
    "bloom_single_key.c",
    "bloom_unified.c",
    "bloom_batched.c",
]

# (filter size key, expected max FP rate)
SIZE_LIMITS = [
    ("S",  0.005),
    ("M",  0.005),
    ("L",  0.005),
]


def run_one(src, size_key, fp_limit):
    sz = harness.BENCH_SIZES[size_key]
    c = harness.compile_candidate(Path(src), Path("runs/test"))
    d = harness.diff_test(
        c,
        nbits=sz["nbits"],
        n_insert=sz["n_insert"],
        n_query=min(sz["n_query"], 200_000),
        fp_limit=fp_limit,
    )
    return c, d


def main():
    failed = 0
    total = 0
    for src in CANDIDATES:
        if not Path(src).exists():
            print(f"SKIP  {src}: not found")
            continue
        for size_key, fp_limit in SIZE_LIMITS:
            total += 1
            try:
                c, d = run_one(src, size_key, fp_limit)
            except Exception as e:
                print(f"FAIL  {src} [{size_key}]: {e}")
                failed += 1
                continue
            status = "PASS" if d.passed else "FAIL"
            note = f"fn={d.false_negatives} fp_rate={d.fp_rate:.5f}"
            if not d.passed:
                note += f" ({d.reason})"
                failed += 1
            print(f"{status}  {Path(src).stem:<20s} [{size_key}]  {note}")
    print()
    print(f"{total - failed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
