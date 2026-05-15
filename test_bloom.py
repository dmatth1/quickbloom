#!/usr/bin/env python3
"""Correctness tests for the Bloom filter implementations.

Compiles each candidate and runs:
  - The fixed-16-byte differential test across cache regimes
    (zero false negatives; FP rate within spec).
  - A variable-length key test: keys of mixed lengths 1..64 bytes,
    inserted via the byte API; verify zero false negatives.
"""
import ctypes
import hashlib
import sys
from pathlib import Path

import harness


CANDIDATES = [
    "bloom_single_key.c",
    "bloom_unified.c",
    "bloom_batched.c",
    "bloom_classic.c",
]

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


def make_varlen_keys(n, seed):
    """Deterministic keys of mixed lengths in 1..64 bytes.

    Returns: list of (offset, length) pointing into a packed buffer, plus
    the buffer bytes.
    """
    base = seed.to_bytes(8, "little")
    parts = []
    for i in range(n):
        h = hashlib.sha256(base + i.to_bytes(8, "little")).digest()
        # length cycles 1..64 deterministically from the first hash byte
        length = (h[0] % 64) + 1
        # repeat hash if length > 32 (sha256 digest is 32 bytes)
        body = (h + hashlib.sha256(h).digest())[:length]
        parts.append(body)
    return parts


def test_varlen(src):
    c = harness.compile_candidate(Path(src), Path("runs/test"))
    nbits = 1 << 22  # 4 Mbits = 512 KB, plenty for 5000 keys
    n = 5000
    bf = c.lib.bloom_new(nbits)
    try:
        inserted = make_varlen_keys(n, seed=101)
        for key in inserted:
            c.lib.bloom_insert(bf, key, len(key))
        # All inserted keys must still be reported as present.
        fn = 0
        for key in inserted:
            if not c.lib.bloom_contains(bf, key, len(key)):
                fn += 1
        # Sanity check on FP rate too: query unseen variable-length keys.
        unseen = make_varlen_keys(n, seed=202)
        fp = sum(1 for key in unseen if c.lib.bloom_contains(bf, key, len(key)))
        fp_rate = fp / n
        return fn, fp_rate
    finally:
        c.lib.bloom_free(bf)


def main():
    failed = 0
    total = 0

    # Phase 1: fixed-length differential test across cache regimes.
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
            print(f"{status}  {Path(src).stem:<20s} [{size_key:<2}]   {note}")

    print()
    # Phase 2: variable-length keys exercise the fasthash64_var path.
    for src in CANDIDATES:
        if not Path(src).exists():
            continue
        total += 1
        try:
            fn, fp_rate = test_varlen(src)
        except Exception as e:
            print(f"FAIL  {src} [varlen]: {e}")
            failed += 1
            continue
        passed = (fn == 0) and (fp_rate < 0.05)  # loose FP cap, this is small-n
        if not passed: failed += 1
        status = "PASS" if passed else "FAIL"
        print(f"{status}  {Path(src).stem:<20s} [varlen] fn={fn} fp_rate={fp_rate:.5f}")

    print()
    print(f"{total - failed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
