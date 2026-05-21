// xxh64.h -- single-shot XXH64, header-only. Vendored from the
// canonical xxHash reference (Yann Collet, BSD-2). Used by
// tools/bench_hash.c (in-binary, static) and tools/xxh64_helper.c
// (exposed as a .so symbol for test/test_compat_arrow_rs.py).
//
// Parquet mandates XXH64 with seed=0; arrow-rs / arrow-cpp / Velox /
// DuckDB / Impala all use that. Two consumers, one implementation,
// no drift.

#ifndef QB_XXH64_H
#define QB_XXH64_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

static const uint64_t QB_XXH_PRIME64_1 = 11400714785074694791ULL;
static const uint64_t QB_XXH_PRIME64_2 = 14029467366897019727ULL;
static const uint64_t QB_XXH_PRIME64_3 = 1609587929392839161ULL;
static const uint64_t QB_XXH_PRIME64_4 = 9650029242287828579ULL;
static const uint64_t QB_XXH_PRIME64_5 = 2870177450012600261ULL;

static inline uint64_t qb_xxh_rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}
static inline uint64_t qb_xxh_read64(const void* p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint32_t qb_xxh_read32(const void* p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static inline uint64_t qb_xxh_round(uint64_t acc, uint64_t input) {
    acc += input * QB_XXH_PRIME64_2;
    acc = qb_xxh_rotl64(acc, 31);
    acc *= QB_XXH_PRIME64_1;
    return acc;
}
static inline uint64_t qb_xxh_merge_round(uint64_t acc, uint64_t val) {
    val = qb_xxh_round(0, val);
    acc ^= val;
    return acc * QB_XXH_PRIME64_1 + QB_XXH_PRIME64_4;
}

static inline uint64_t qb_xxh64(const void* input, size_t len, uint64_t seed) {
    const uint8_t* p = (const uint8_t*)input;
    const uint8_t* const end = p + len;
    uint64_t h64;

    if (len >= 32) {
        const uint8_t* const limit = end - 32;
        uint64_t v1 = seed + QB_XXH_PRIME64_1 + QB_XXH_PRIME64_2;
        uint64_t v2 = seed + QB_XXH_PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - QB_XXH_PRIME64_1;
        do {
            v1 = qb_xxh_round(v1, qb_xxh_read64(p)); p += 8;
            v2 = qb_xxh_round(v2, qb_xxh_read64(p)); p += 8;
            v3 = qb_xxh_round(v3, qb_xxh_read64(p)); p += 8;
            v4 = qb_xxh_round(v4, qb_xxh_read64(p)); p += 8;
        } while (p <= limit);
        h64 = qb_xxh_rotl64(v1, 1) + qb_xxh_rotl64(v2, 7)
            + qb_xxh_rotl64(v3, 12) + qb_xxh_rotl64(v4, 18);
        h64 = qb_xxh_merge_round(h64, v1);
        h64 = qb_xxh_merge_round(h64, v2);
        h64 = qb_xxh_merge_round(h64, v3);
        h64 = qb_xxh_merge_round(h64, v4);
    } else {
        h64 = seed + QB_XXH_PRIME64_5;
    }
    h64 += (uint64_t)len;
    while (p + 8 <= end) {
        uint64_t k1 = qb_xxh_round(0, qb_xxh_read64(p));
        h64 ^= k1;
        h64 = qb_xxh_rotl64(h64, 27) * QB_XXH_PRIME64_1 + QB_XXH_PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h64 ^= (uint64_t)qb_xxh_read32(p) * QB_XXH_PRIME64_1;
        h64 = qb_xxh_rotl64(h64, 23) * QB_XXH_PRIME64_2 + QB_XXH_PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h64 ^= (uint64_t)(*p) * QB_XXH_PRIME64_5;
        h64 = qb_xxh_rotl64(h64, 11) * QB_XXH_PRIME64_1;
        p++;
    }
    h64 ^= h64 >> 33;
    h64 *= QB_XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= QB_XXH_PRIME64_3;
    h64 ^= h64 >> 32;
    return h64;
}

#endif // QB_XXH64_H
