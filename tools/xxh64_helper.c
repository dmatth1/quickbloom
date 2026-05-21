// xxh64_helper.c -- single-shot XXH64 exposed as a shared-library
// symbol, for the cross-validation test
// (test/test_compat_arrow_rs.py) that needs to feed quickbloom and
// arrow-rs the SAME 64-bit hash so the bitsets are directly
// comparable. Parquet mandates XXH64 with seed=0; arrow-rs computes
// it internally via the `twox-hash` crate; we vendor the canonical
// C implementation in tools/xxh64.h so the test doesn't require any
// extra Python or Rust dependency.

#include "xxh64.h"

uint64_t xxh64(const void* input, size_t len, uint64_t seed) {
    return qb_xxh64(input, len, seed);
}
