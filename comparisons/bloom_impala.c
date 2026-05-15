// bloom_impala.c -- Apache Parquet / Impala SBBF reference (scalar).
//
// Faithful implementation of the SBBF spec (parquet-format/BloomFilter.md)
// as shipped in Apache Parquet, Apache Impala, and Apache Arrow. Used as
// an external competitor in this repo's benchmark.
//
// Layout:
//   - Block = 256 bits = eight 32-bit words = half a cache line.
//   - Per insert: pick one block from the 64-bit key hash (upper 32 bits,
//     via Lemire's fastrange to avoid mod), then set one bit in each of
//     the 8 words. The bit position within word i is (h32 * SALT[i]) >> 27,
//     yielding a value in [0,31].
//   - Eight specific odd 32-bit salt constants from the Parquet spec.
//   - Per-key hash: XXH64 with seed 0 (Parquet's choice).
//
// vs `bloom_single_key.c` in this repo: same SBBF algorithm and same Parquet
// salts, but this file uses XXH64 (per spec) and scalar bit-setting (no
// SIMD mask compute). It is what a "by-the-book" Parquet SBBF would do
// from a clean read of the spec, and is comparable to Impala's production
// SBBF on the single-key API.

#define K_HASHES 8

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// --- XXH64 (faithful to xxHash spec, scalar) -------------------------------
#define XXH_PRIME64_1 0x9E3779B185EBCA87ULL
#define XXH_PRIME64_2 0xC2B2AE3D27D4EB4FULL
#define XXH_PRIME64_3 0x165667B19E3779F9ULL
#define XXH_PRIME64_4 0x85EBCA77C2B2AE63ULL
#define XXH_PRIME64_5 0x27D4EB2F165667C5ULL

static inline uint64_t xxh_rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t xxh_round(uint64_t acc, uint64_t input) {
    acc += input * XXH_PRIME64_2;
    acc = xxh_rotl64(acc, 31);
    return acc * XXH_PRIME64_1;
}

static inline uint64_t xxh_merge_round(uint64_t acc, uint64_t val) {
    val = xxh_round(0, val);
    acc ^= val;
    return acc * XXH_PRIME64_1 + XXH_PRIME64_4;
}

static uint64_t xxh64(const void* data, size_t len, uint64_t seed) {
    const uint8_t* p = (const uint8_t*)data;
    const uint8_t* end = p + len;
    uint64_t h;

    if (len >= 32) {
        uint64_t v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
        uint64_t v2 = seed + XXH_PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - XXH_PRIME64_1;
        const uint8_t* limit = end - 32;
        do {
            uint64_t k;
            memcpy(&k, p, 8);     v1 = xxh_round(v1, k); p += 8;
            memcpy(&k, p, 8);     v2 = xxh_round(v2, k); p += 8;
            memcpy(&k, p, 8);     v3 = xxh_round(v3, k); p += 8;
            memcpy(&k, p, 8);     v4 = xxh_round(v4, k); p += 8;
        } while (p <= limit);

        h = xxh_rotl64(v1, 1)  + xxh_rotl64(v2, 7)
          + xxh_rotl64(v3, 12) + xxh_rotl64(v4, 18);
        h = xxh_merge_round(h, v1);
        h = xxh_merge_round(h, v2);
        h = xxh_merge_round(h, v3);
        h = xxh_merge_round(h, v4);
    } else {
        h = seed + XXH_PRIME64_5;
    }

    h += (uint64_t)len;

    while (p + 8 <= end) {
        uint64_t k;
        memcpy(&k, p, 8);
        h ^= xxh_round(0, k);
        h = xxh_rotl64(h, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        uint32_t k;
        memcpy(&k, p, 4);
        h ^= (uint64_t)k * XXH_PRIME64_1;
        h = xxh_rotl64(h, 23) * XXH_PRIME64_2 + XXH_PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * XXH_PRIME64_5;
        h = xxh_rotl64(h, 11) * XXH_PRIME64_1;
        p++;
    }

    h ^= h >> 33;
    h *= XXH_PRIME64_2;
    h ^= h >> 29;
    h *= XXH_PRIME64_3;
    h ^= h >> 32;
    return h;
}

// --- SBBF -------------------------------------------------------------------
#define SBBF_WORDS 8
#define SBBF_BLOCK_BYTES 32

static const uint32_t SBBF_SALT[8] = {
    0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
    0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U,
};

typedef struct {
    uint32_t* bits;     // nblocks * 8 uint32_t, 32-byte aligned
    size_t nblocks;
} bloom_t;

static inline void block_mask(uint32_t h32, uint32_t out[8]) {
    for (int i = 0; i < 8; i++) {
        uint32_t bit = (h32 * SBBF_SALT[i]) >> 27;
        out[i] = 1u << bit;
    }
}

static inline uint32_t* block_for(const bloom_t* b, uint64_t h64) {
    uint32_t hi = (uint32_t)(h64 >> 32);
    size_t idx = ((uint64_t)hi * (uint64_t)b->nblocks) >> 32;
    return b->bits + idx * SBBF_WORDS;
}

void* bloom_new(size_t nbits) {
    size_t nblocks = (nbits + 255) / 256;
    if (nblocks == 0) nblocks = 1;
    bloom_t* b = (bloom_t*)malloc(sizeof(bloom_t));
    b->nblocks = nblocks;
    size_t nbytes = nblocks * SBBF_BLOCK_BYTES;
    // Round up to multiple of 32 (already is) and 32-byte-align for AVX2.
    void* mem = NULL;
    if (posix_memalign(&mem, 32, nbytes) != 0) mem = calloc(nbytes, 1);
    else memset(mem, 0, nbytes);
    b->bits = (uint32_t*)mem;
    return b;
}

void bloom_free(void* p) {
    bloom_t* b = (bloom_t*)p;
    free(b->bits);
    free(b);
}

void bloom_insert(void* p, const void* key, size_t len) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h = xxh64(key, len, 0);
    uint32_t* blk = block_for(b, h);
    uint32_t mask[8];
    block_mask((uint32_t)h, mask);
    for (int i = 0; i < 8; i++) blk[i] |= mask[i];
}

int bloom_contains(void* p, const void* key, size_t len) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h = xxh64(key, len, 0);
    uint32_t* blk = block_for(b, h);
    uint32_t mask[8];
    block_mask((uint32_t)h, mask);
    for (int i = 0; i < 8; i++) {
        if ((blk[i] & mask[i]) != mask[i]) return 0;
    }
    return 1;
}

void bloom_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    for (size_t i = 0; i < n; i++) {
        bloom_insert(p, keys + i * klen, klen);
    }
}

size_t bloom_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        if (bloom_contains(p, keys + i * klen, klen)) hits++;
    }
    return hits;
}

// Prehashed ABI -- same algorithm, hash done outside the timed loop.
void bloom_insert_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint32_t* blk = block_for(b, h);
    uint32_t mask[8];
    block_mask((uint32_t)h, mask);
    for (int i = 0; i < 8; i++) blk[i] |= mask[i];
}

int bloom_contains_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint32_t* blk = block_for(b, h);
    uint32_t mask[8];
    block_mask((uint32_t)h, mask);
    for (int i = 0; i < 8; i++) {
        if ((blk[i] & mask[i]) != mask[i]) return 0;
    }
    return 1;
}

void bloom_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    for (size_t i = 0; i < n; i++) bloom_insert_prehash(p, hashes[i]);
}

size_t bloom_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        if (bloom_contains_prehash(p, hashes[i])) hits++;
    }
    return hits;
}
