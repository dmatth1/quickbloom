// xxh64_helper.c -- single-shot XXH64 exposed as a shared-library
// symbol, for the cross-validation test (test/test_compat_arrow_rs.py)
// that needs to feed quickbloom and arrow-rs the SAME 64-bit hash so
// the bitsets are directly comparable. Parquet mandates XXH64 with
// seed=0; arrow-rs computes it internally via the `twox-hash` crate;
// we vendor the canonical C implementation here so the test doesn't
// require any extra Python or Rust dependency.
//
// Algorithm: xxHash 64-bit, single-shot, no streaming context. Same
// code path as tools/bench_hash.c's xxh64 (kept in sync there).
// BSD-2-Clause, Yann Collet.

#include <stdint.h>
#include <stddef.h>
#include <string.h>

static const uint64_t XXH_PRIME64_1 = 11400714785074694791ULL;
static const uint64_t XXH_PRIME64_2 = 14029467366897019727ULL;
static const uint64_t XXH_PRIME64_3 = 1609587929392839161ULL;
static const uint64_t XXH_PRIME64_4 = 9650029242287828579ULL;
static const uint64_t XXH_PRIME64_5 = 2870177450012600261ULL;

static inline uint64_t xxh_rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}
static inline uint64_t xxh_read64(const void* p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t xxh_read32(const void* p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static inline uint64_t xxh_round(uint64_t acc, uint64_t input) {
    acc += input * XXH_PRIME64_2;
    acc = xxh_rotl64(acc, 31);
    acc *= XXH_PRIME64_1;
    return acc;
}
static inline uint64_t xxh_merge_round(uint64_t acc, uint64_t val) {
    val = xxh_round(0, val);
    acc ^= val;
    return acc * XXH_PRIME64_1 + XXH_PRIME64_4;
}

uint64_t xxh64(const void* input, size_t len, uint64_t seed) {
    const uint8_t* p = (const uint8_t*)input;
    const uint8_t* const end = p + len;
    uint64_t h64;

    if (len >= 32) {
        const uint8_t* const limit = end - 32;
        uint64_t v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
        uint64_t v2 = seed + XXH_PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - XXH_PRIME64_1;
        do {
            v1 = xxh_round(v1, xxh_read64(p)); p += 8;
            v2 = xxh_round(v2, xxh_read64(p)); p += 8;
            v3 = xxh_round(v3, xxh_read64(p)); p += 8;
            v4 = xxh_round(v4, xxh_read64(p)); p += 8;
        } while (p <= limit);
        h64 = xxh_rotl64(v1, 1) + xxh_rotl64(v2, 7) + xxh_rotl64(v3, 12) + xxh_rotl64(v4, 18);
        h64 = xxh_merge_round(h64, v1);
        h64 = xxh_merge_round(h64, v2);
        h64 = xxh_merge_round(h64, v3);
        h64 = xxh_merge_round(h64, v4);
    } else {
        h64 = seed + XXH_PRIME64_5;
    }
    h64 += (uint64_t)len;
    while (p + 8 <= end) {
        uint64_t k1 = xxh_round(0, xxh_read64(p));
        h64 ^= k1;
        h64 = xxh_rotl64(h64, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h64 ^= (uint64_t)xxh_read32(p) * XXH_PRIME64_1;
        h64 = xxh_rotl64(h64, 23) * XXH_PRIME64_2 + XXH_PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * XXH_PRIME64_5;
        h64 = xxh_rotl64(h64, 11) * XXH_PRIME64_1;
        p++;
    }
    h64 ^= h64 >> 33;
    h64 *= XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= XXH_PRIME64_3;
    h64 ^= h64 >> 32;
    return h64;
}
