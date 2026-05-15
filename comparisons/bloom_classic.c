// bloom_classic.c -- textbook non-blocked Bloom filter, baseline competitor.
//
// This is the canonical pre-SBBF design that the rest of this repo is
// trying to beat: K independent bit positions across the whole bit array,
// no blocking, no SIMD. Same wymum hash as the SBBF variants so any
// performance gap is attributable to the data structure, not the hash.
//
// Why include it: published Bloom filter implementations (Java, Cassandra
// pre-3.x, classic textbook code) use this layout. Without a baseline,
// the "X cyc/op" numbers for SBBF have no anchor.
//
// Algorithm: double-hashing -- one 128-bit wymum gives (h1, h2), then
// bit_i = (h1 + i*h2) mod nbits. K=8 by default to match single_key.c.
// Power-of-2 nbits so mod is a mask.
//
// Expected to lose: at K=8 every contains() may touch up to 8 distinct
// cache lines. Out-of-cache this means ~8x more DRAM round-trips than
// SBBF, which touches one block per query.

#define K_HASHES 8

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline void hash16_pair(const void* data, uint64_t* h1, uint64_t* h2) {
    uint64_t a, b;
    memcpy(&a, data, 8);
    memcpy(&b, (const uint8_t*)data + 8, 8);
    __uint128_t r = (__uint128_t)a * b;
    *h1 = (uint64_t)r ^ (uint64_t)(r >> 64);
    // Mix a second derived word for the double-hashing pair.
    uint64_t m = *h1 * 0x9e3779b97f4a7c15ULL;
    m ^= m >> 32;
    *h2 = m | 1;  // odd so adds keep cycling through residues
}

static inline uint64_t fasthash64_var(const void* data, size_t len) {
    const uint64_t M = 0x880355f21e6d1965ULL;
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = (uint64_t)len * M;
    size_t nblocks = len / 8;
    for (size_t i = 0; i < nblocks; i++) {
        uint64_t k;
        memcpy(&k, p + i * 8, 8);
        k *= M; k ^= k >> 23; k *= M;
        h ^= k; h *= M;
    }
    const uint8_t* tail = p + nblocks * 8;
    uint64_t t = 0;
    switch (len & 7) {
        case 7: t ^= (uint64_t)tail[6] << 48; /* fall through */
        case 6: t ^= (uint64_t)tail[5] << 40;
        case 5: t ^= (uint64_t)tail[4] << 32;
        case 4: t ^= (uint64_t)tail[3] << 24;
        case 3: t ^= (uint64_t)tail[2] << 16;
        case 2: t ^= (uint64_t)tail[1] << 8;
        case 1: t ^= (uint64_t)tail[0];
                t *= M; t ^= t >> 23; t *= M; h ^= t; h *= M;
    }
    h ^= h >> 23; h *= 0x2127599bf4325c37ULL; h ^= h >> 47;
    return h;
}

static inline void bloom_hash_pair(const void* data, size_t len,
                                   uint64_t* h1, uint64_t* h2) {
    if (__builtin_expect(len == 16, 1)) { hash16_pair(data, h1, h2); return; }
    uint64_t h = fasthash64_var(data, len);
    *h1 = h;
    uint64_t m = h * 0x9e3779b97f4a7c15ULL;
    m ^= m >> 32;
    *h2 = m | 1;
}

typedef struct {
    uint64_t* bits;          // bit array, nwords = nbits / 64
    size_t bit_mask;         // nbits - 1 (power-of-2 bits)
} bloom_t;

void* bloom_new(size_t nbits) {
    size_t pow2 = 1;
    while (pow2 < nbits) pow2 <<= 1;
    nbits = pow2;
    size_t nwords = nbits / 64;
    if (nwords == 0) { nwords = 1; nbits = 64; }
    bloom_t* b = (bloom_t*)malloc(sizeof(bloom_t));
    b->bit_mask = nbits - 1;
    void* mem = NULL;
    if (posix_memalign(&mem, 64, nwords * 8) != 0) mem = calloc(nwords, 8);
    else memset(mem, 0, nwords * 8);
    b->bits = (uint64_t*)mem;
    return b;
}

void bloom_free(void* p) {
    bloom_t* b = (bloom_t*)p;
    free(b->bits);
    free(b);
}

static inline void set_bit(bloom_t* b, uint64_t pos) {
    pos &= b->bit_mask;
    b->bits[pos >> 6] |= (1ULL << (pos & 63));
}

static inline int get_bit(const bloom_t* b, uint64_t pos) {
    pos &= b->bit_mask;
    return (b->bits[pos >> 6] >> (pos & 63)) & 1ULL;
}

static inline void insert_with_pair(bloom_t* b, uint64_t h1, uint64_t h2) {
    uint64_t pos = h1;
    for (int i = 0; i < K_HASHES; i++) {
        set_bit(b, pos);
        pos += h2;
    }
}

static inline int contains_with_pair(const bloom_t* b, uint64_t h1, uint64_t h2) {
    uint64_t pos = h1;
    for (int i = 0; i < K_HASHES; i++) {
        if (!get_bit(b, pos)) return 0;
        pos += h2;
    }
    return 1;
}

void bloom_insert(void* p, const void* key, size_t len) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h1, h2;
    bloom_hash_pair(key, len, &h1, &h2);
    insert_with_pair(b, h1, h2);
}

int bloom_contains(void* p, const void* key, size_t len) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h1, h2;
    bloom_hash_pair(key, len, &h1, &h2);
    return contains_with_pair(b, h1, h2);
}

void bloom_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    for (size_t i = 0; i < n; i++) bloom_insert(p, keys + i * klen, klen);
}

size_t bloom_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        if (bloom_contains(p, keys + i * klen, klen)) hits++;
    }
    return hits;
}

// Prehashed ABI: caller hands us h1; we derive h2 deterministically so
// the FP-rate test stays comparable across runs.
static inline void prehash_split(uint64_t h, uint64_t* h1, uint64_t* h2) {
    *h1 = h;
    uint64_t m = h * 0x9e3779b97f4a7c15ULL;
    m ^= m >> 32;
    *h2 = m | 1;
}

void bloom_insert_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h1, h2;
    prehash_split(h, &h1, &h2);
    insert_with_pair(b, h1, h2);
}

int bloom_contains_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h1, h2;
    prehash_split(h, &h1, &h2);
    return contains_with_pair(b, h1, h2);
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
