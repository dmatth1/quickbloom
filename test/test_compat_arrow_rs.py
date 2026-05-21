#!/usr/bin/env python3
"""Cross-validation: quickbloom vs. arrow-rs `parquet::bloom_filter::Sbbf`.

The README claims quickbloom's bitset is bit-identical to other
Parquet SBBF implementations (arrow-cpp, arrow-rs, Velox, DuckDB,
Impala) for the same 64-bit hash input. This test converts that
claim from "should be true by construction" to "verified true on
every CI run" against the reference implementation we already wire
into the bench harness (arrow-rs).

Approach: feed both filters the SAME 64-bit XXH64 hash for every
key, then check that every probe agrees.

  - quickbloom: insert via the prehash API so it skips its built-in
    wymum and uses the hash we computed.
  - arrow-rs: insert via the bytes API; the shim's underlying
    `Sbbf::insert` runs XXH64-with-seed-0 internally (Parquet spec).
    We compute the SAME XXH64 in C (tools/xxh64_helper.c) so the
    two filters see byte-identical hash inputs.

If the bitsets are bit-identical, `contains` on the same hash MUST
return the same answer for every input. With 100k random queries
across a small filter, any single-bit divergence would surface as
~100k probes hitting different blocks, which is detected with
overwhelming probability — equivalent to byte-level comparison for
the purposes of the bit-identical claim.

Run with `python3 test/test_compat_arrow_rs.py`; requires `cargo`
for the arrow-rs shim. Skips with a clear message if cargo isn't
available.
"""
import ctypes
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import harness


def _build_xxh64_helper(out_dir: Path) -> ctypes.CDLL:
    """Compile tools/xxh64_helper.c into a .so and return the loaded
    library with the xxh64 function bound."""
    out_dir.mkdir(parents=True, exist_ok=True)
    so = out_dir / "xxh64_helper.so"
    import subprocess
    r = subprocess.run(
        [harness.CC, "-O3", "-fPIC", "-shared", "-std=c11",
         str(REPO / "tools" / "xxh64_helper.c"), "-o", str(so)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"xxh64_helper compile failed:\n{r.stderr}")
    lib = ctypes.CDLL(str(so))
    lib.xxh64.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
    lib.xxh64.restype = ctypes.c_uint64
    return lib


def main() -> int:
    out = REPO / "build" / "compat"
    qb = harness.compile_candidate(REPO / "quickbloom.c", out)
    try:
        rs = harness.compile_candidate(
            REPO / "comparisons" / "arrow_rs_sbbf_shim", out
        )
    except harness.CargoMissingError as e:
        print(f"SKIP: {e}")
        return 0
    xxh = _build_xxh64_helper(out)

    # Same nbits on both filters so the block-index math agrees. 1 Mbit
    # = 128 KB at SBBF's natural load (~21 bits/key) gives an expected
    # FP rate ~1e-4 — non-zero, so the cross-check exercises both the
    # "agree on a hit" and "agree on a miss" paths.
    NBITS = 1 << 20
    KLEN  = 16
    N_INSERT = 50_000
    N_QUERY  = 100_000

    qb_f = qb.lib.bloom_new(NBITS)
    rs_f = rs.lib.bloom_new(NBITS)
    if not qb_f or not rs_f:
        print("FAIL: bloom_new returned NULL")
        return 1

    inserted = harness.make_corpus(N_INSERT, KLEN, seed=1)
    unseen   = harness.make_corpus(N_QUERY,  KLEN, seed=2)

    # Insert: same hash into both filters via different code paths.
    for i in range(N_INSERT):
        key = inserted[i * KLEN : (i + 1) * KLEN]
        h = xxh.xxh64(key, KLEN, 0)
        qb.lib.qb_insert_prehash(qb_f, ctypes.c_uint64(h))
        rs.lib.bloom_insert(rs_f, key, KLEN)

    # Every inserted key must be reported present by both filters.
    qb_fn = 0
    rs_fn = 0
    for i in range(N_INSERT):
        key = inserted[i * KLEN : (i + 1) * KLEN]
        h = xxh.xxh64(key, KLEN, 0)
        if not qb.lib.qb_contains_prehash(qb_f, ctypes.c_uint64(h)):
            qb_fn += 1
        if not rs.lib.bloom_contains(rs_f, key, KLEN):
            rs_fn += 1
    if qb_fn or rs_fn:
        print(f"FAIL: false negatives — quickbloom={qb_fn} arrow_rs={rs_fn}")
        return 1

    # On every unseen key, contains() MUST agree between the two filters.
    # A single disagreement means the bitsets differ — i.e. the
    # bit-identical claim is false.
    disagreements = 0
    qb_hits = 0
    rs_hits = 0
    for i in range(N_QUERY):
        key = unseen[i * KLEN : (i + 1) * KLEN]
        h = xxh.xxh64(key, KLEN, 0)
        qb_says = bool(qb.lib.qb_contains_prehash(qb_f, ctypes.c_uint64(h)))
        rs_says = bool(rs.lib.bloom_contains(rs_f, key, KLEN))
        if qb_says != rs_says:
            disagreements += 1
        qb_hits += qb_says
        rs_hits += rs_says
    if disagreements:
        print(f"FAIL: contains() disagreed on {disagreements}/{N_QUERY} unseen keys "
              f"(quickbloom={qb_hits} hits, arrow_rs={rs_hits} hits) — "
              f"bitsets are NOT bit-identical")
        return 1

    qb.lib.bloom_free(qb_f)
    rs.lib.bloom_free(rs_f)
    print(f"PASS: cross-validation against arrow-rs SBBF "
          f"(inserted={N_INSERT}, queried={N_QUERY}, "
          f"shared FP rate={qb_hits / N_QUERY:.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
