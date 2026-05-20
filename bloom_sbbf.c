// bloom_sbbf.c -- SBBF Bloom filter, single-key API.
//
// One implementation, two PREFETCH_LOOKAHEAD configurations, three
// possible link-time namespaces. Compile via one of the variant entry
// points (bloom_single_key.c, bloom_unified.c) or directly with
// `-DQB_NS=qb_<variant> -DPREFETCH_LOOKAHEAD={0,8}`.
//
// Split Block Bloom Filter (Apache Parquet spec):
//   - 256-bit blocks (8 x 32-bit words), K=8 (one bit per word)
//   - wymum hash (single 128-bit multiply) on 16-byte fast path,
//     fasthash64 fallback for variable-length keys
//   - Power-of-2 nblocks with bitmask block-index (no fastrange)
//   - AVX2 SIMD mask compute (vpmullo + vpsrli + vpsllv)
//   - 4-way unrolled bulk paths
//   - bulk_insert uses sequential load+OR+store per key so hardware
//     store-to-load forwarding handles aliased blocks correctly
//
// Requirements: x86_64 with AVX2 + BMI2.

#ifndef QB_NS
#  error "bloom_sbbf.c requires QB_NS to be defined (e.g. -DQB_NS=qb_single_key). Compile via bloom_single_key.c or bloom_unified.c."
#endif

// quickbloom requires x86_64 with AVX2 + BMI2. Catch ports to other
// architectures (ARM, RISC-V, 32-bit x86) at the top of the file with a
// clear error rather than a wall of confusing intrinsic-not-found
// messages from immintrin.h.
#if !defined(__x86_64__) && !defined(_M_X64)
#  error "quickbloom requires x86_64"
#endif
#if !defined(__AVX2__)
#  error "quickbloom requires AVX2 — compile with at least -mavx2 -mbmi2"
#endif

#ifndef PREFETCH_LOOKAHEAD
#  define PREFETCH_LOOKAHEAD 0
#endif

#ifndef K_HASHES
#  define K_HASHES 8
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

// Namespace plumbing — every public function is named via QB_API(...)
// so the same source produces qb_single_key_* or qb_unified_* symbols
// depending on the QB_NS macro at compile time.
#define QB_CONCAT2(a, b) a##_##b
#define QB_CONCAT(a, b) QB_CONCAT2(a, b)
#define QB_API(name) QB_CONCAT(QB_NS, name)

static inline uint64_t hash16(const void* data) {
    uint64_t a, b;
    memcpy(&a, data, 8);
    memcpy(&b, (const uint8_t*)data + 8, 8);
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
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
        case 7: t ^= (uint64_t)tail[6] << 48; __attribute__((fallthrough));
        case 6: t ^= (uint64_t)tail[5] << 40; __attribute__((fallthrough));
        case 5: t ^= (uint64_t)tail[4] << 32; __attribute__((fallthrough));
        case 4: t ^= (uint64_t)tail[3] << 24; __attribute__((fallthrough));
        case 3: t ^= (uint64_t)tail[2] << 16; __attribute__((fallthrough));
        case 2: t ^= (uint64_t)tail[1] << 8;  __attribute__((fallthrough));
        case 1: t ^= (uint64_t)tail[0];
                t *= M; t ^= t >> 23; t *= M; h ^= t; h *= M;
    }
    h ^= h >> 23; h *= 0x2127599bf4325c37ULL; h ^= h >> 47;
    return h;
}

static inline uint64_t bloom_hash(const void* data, size_t len) {
    if (__builtin_expect(len == 16, 1)) return hash16(data);
    return fasthash64_var(data, len);
}

#define SBBF_BLOCK_BYTES 32

typedef struct {
    uint32_t* bits;
    size_t nblocks_mask;
} bloom_t;

static inline __m256i mask_for(uint32_t h32) {
    const __m256i salt = _mm256_set_epi32(
        (int)0x5c6bfb31u, (int)0x9efc4947u, (int)0x2df1424bu, (int)0x705495c7u,
        (int)0xa2b7289du, (int)0x8824ad5bu, (int)0x44974d91u, (int)0x47b6137bu);
    const __m256i hbcast = _mm256_set1_epi32((int)h32);
    const __m256i prod = _mm256_mullo_epi32(hbcast, salt);
    const __m256i shift = _mm256_srli_epi32(prod, 27);
    const __m256i ones = _mm256_set1_epi32(1);
    return _mm256_sllv_epi32(ones, shift);
}

static inline uint32_t* block_for(const bloom_t* b, uint64_t h64) {
    size_t idx = ((size_t)(h64 >> 32)) & b->nblocks_mask;
    return b->bits + idx * 8;
}

#if PREFETCH_LOOKAHEAD > 0
static inline void prefetch_block_for(const bloom_t* b, uint64_t h64) {
    __builtin_prefetch(block_for(b, h64), 0, 0);
}
#endif

// QB_API(new) returns NULL on allocation failure. The returned pointer
// must be released with QB_API(free).
void* QB_API(new)(size_t nbits) {
    size_t nblocks = (nbits + 255) / 256;
    if (nblocks == 0) nblocks = 1;
    size_t pow2 = 1;
    while (pow2 < nblocks) pow2 <<= 1;
    nblocks = pow2;
    bloom_t* b = (bloom_t*)malloc(sizeof(bloom_t));
    if (!b) return NULL;
    size_t nbytes = nblocks * SBBF_BLOCK_BYTES;
    void* mem = NULL;
    // posix_memalign guarantees 32-byte alignment required by _mm256_load.
    // We do not fall back to calloc on failure because calloc cannot
    // satisfy the alignment contract; failing fast is safer than
    // returning a filter that will SEGV on the first insert.
    if (posix_memalign(&mem, 32, nbytes) != 0) {
        free(b);
        return NULL;
    }
    memset(mem, 0, nbytes);
    b->nblocks_mask = nblocks - 1;
    b->bits = (uint32_t*)mem;
    return b;
}

void QB_API(free)(void* p) {
    if (!p) return;
    bloom_t* b = (bloom_t*)p;
    free(b->bits);
    free(b);
}

void QB_API(insert)(void* p, const void* key, size_t len) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h = bloom_hash(key, len);
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    _mm256_store_si256((__m256i*)blk, _mm256_or_si256(cur, m));
}

int QB_API(contains)(void* p, const void* key, size_t len) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h = bloom_hash(key, len);
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    return _mm256_testc_si256(cur, m);
}

#define APPLY(P, M) do {                                          \
    __m256i c_ = _mm256_load_si256((__m256i*)(P));                \
    _mm256_store_si256((__m256i*)(P), _mm256_or_si256(c_, M));    \
} while (0)

void QB_API(insert_bulk)(void* p, const uint8_t* keys, size_t klen, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t i = 0;
    if (klen == 16) {
        for (; i + 4 <= n; i += 4) {
            uint64_t h0 = hash16(keys + (i+0)*16);
            uint64_t h1 = hash16(keys + (i+1)*16);
            uint64_t h2 = hash16(keys + (i+2)*16);
            uint64_t h3 = hash16(keys + (i+3)*16);
#if PREFETCH_LOOKAHEAD > 0
            if (i + PREFETCH_LOOKAHEAD + 4 <= n) {
                uint64_t pf0 = hash16(keys + (i+PREFETCH_LOOKAHEAD+0)*16);
                uint64_t pf1 = hash16(keys + (i+PREFETCH_LOOKAHEAD+1)*16);
                uint64_t pf2 = hash16(keys + (i+PREFETCH_LOOKAHEAD+2)*16);
                uint64_t pf3 = hash16(keys + (i+PREFETCH_LOOKAHEAD+3)*16);
                prefetch_block_for(b, pf0);
                prefetch_block_for(b, pf1);
                prefetch_block_for(b, pf2);
                prefetch_block_for(b, pf3);
            }
#endif
            uint32_t* p0 = block_for(b, h0);
            uint32_t* p1 = block_for(b, h1);
            uint32_t* p2 = block_for(b, h2);
            uint32_t* p3 = block_for(b, h3);
            __m256i m0 = mask_for((uint32_t)h0);
            __m256i m1 = mask_for((uint32_t)h1);
            __m256i m2 = mask_for((uint32_t)h2);
            __m256i m3 = mask_for((uint32_t)h3);
            APPLY(p0, m0); APPLY(p1, m1); APPLY(p2, m2); APPLY(p3, m3);
        }
    }
    for (; i < n; i++) QB_API(insert)(b, keys + i * klen, klen);
}

size_t QB_API(contains_bulk)(void* p, const uint8_t* keys, size_t klen, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t hits = 0;
    size_t i = 0;
    if (klen == 16) {
        for (; i + 4 <= n; i += 4) {
            uint64_t h0 = hash16(keys + (i+0)*16);
            uint64_t h1 = hash16(keys + (i+1)*16);
            uint64_t h2 = hash16(keys + (i+2)*16);
            uint64_t h3 = hash16(keys + (i+3)*16);
#if PREFETCH_LOOKAHEAD > 0
            if (i + PREFETCH_LOOKAHEAD + 4 <= n) {
                uint64_t pf0 = hash16(keys + (i+PREFETCH_LOOKAHEAD+0)*16);
                uint64_t pf1 = hash16(keys + (i+PREFETCH_LOOKAHEAD+1)*16);
                uint64_t pf2 = hash16(keys + (i+PREFETCH_LOOKAHEAD+2)*16);
                uint64_t pf3 = hash16(keys + (i+PREFETCH_LOOKAHEAD+3)*16);
                prefetch_block_for(b, pf0);
                prefetch_block_for(b, pf1);
                prefetch_block_for(b, pf2);
                prefetch_block_for(b, pf3);
            }
#endif
            uint32_t* p0 = block_for(b, h0);
            uint32_t* p1 = block_for(b, h1);
            uint32_t* p2 = block_for(b, h2);
            uint32_t* p3 = block_for(b, h3);
            __m256i m0 = mask_for((uint32_t)h0);
            __m256i m1 = mask_for((uint32_t)h1);
            __m256i m2 = mask_for((uint32_t)h2);
            __m256i m3 = mask_for((uint32_t)h3);
            __m256i c0 = _mm256_load_si256((__m256i*)p0);
            __m256i c1 = _mm256_load_si256((__m256i*)p1);
            __m256i c2 = _mm256_load_si256((__m256i*)p2);
            __m256i c3 = _mm256_load_si256((__m256i*)p3);
            hits += _mm256_testc_si256(c0, m0);
            hits += _mm256_testc_si256(c1, m1);
            hits += _mm256_testc_si256(c2, m2);
            hits += _mm256_testc_si256(c3, m3);
        }
    }
    for (; i < n; i++) {
        if (QB_API(contains)(b, keys + i * klen, klen)) hits++;
    }
    return hits;
}

void QB_API(insert_prehash)(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    _mm256_store_si256((__m256i*)blk, _mm256_or_si256(cur, m));
}

int QB_API(contains_prehash)(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    return _mm256_testc_si256(cur, m);
}

void QB_API(insert_prehash_bulk)(void* p, const uint64_t* hashes, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
#if PREFETCH_LOOKAHEAD > 0
        if (i + PREFETCH_LOOKAHEAD + 4 <= n) {
            prefetch_block_for(b, hashes[i+PREFETCH_LOOKAHEAD+0]);
            prefetch_block_for(b, hashes[i+PREFETCH_LOOKAHEAD+1]);
            prefetch_block_for(b, hashes[i+PREFETCH_LOOKAHEAD+2]);
            prefetch_block_for(b, hashes[i+PREFETCH_LOOKAHEAD+3]);
        }
#endif
        uint32_t* p0 = block_for(b, hashes[i+0]);
        uint32_t* p1 = block_for(b, hashes[i+1]);
        uint32_t* p2 = block_for(b, hashes[i+2]);
        uint32_t* p3 = block_for(b, hashes[i+3]);
        __m256i m0 = mask_for((uint32_t)hashes[i+0]);
        __m256i m1 = mask_for((uint32_t)hashes[i+1]);
        __m256i m2 = mask_for((uint32_t)hashes[i+2]);
        __m256i m3 = mask_for((uint32_t)hashes[i+3]);
        APPLY(p0, m0); APPLY(p1, m1); APPLY(p2, m2); APPLY(p3, m3);
    }
    for (; i < n; i++) QB_API(insert_prehash)(p, hashes[i]);
}

size_t QB_API(contains_prehash_bulk)(void* p, const uint64_t* hashes, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t hits = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
#if PREFETCH_LOOKAHEAD > 0
        if (i + PREFETCH_LOOKAHEAD + 4 <= n) {
            prefetch_block_for(b, hashes[i+PREFETCH_LOOKAHEAD+0]);
            prefetch_block_for(b, hashes[i+PREFETCH_LOOKAHEAD+1]);
            prefetch_block_for(b, hashes[i+PREFETCH_LOOKAHEAD+2]);
            prefetch_block_for(b, hashes[i+PREFETCH_LOOKAHEAD+3]);
        }
#endif
        uint32_t* p0 = block_for(b, hashes[i+0]);
        uint32_t* p1 = block_for(b, hashes[i+1]);
        uint32_t* p2 = block_for(b, hashes[i+2]);
        uint32_t* p3 = block_for(b, hashes[i+3]);
        __m256i m0 = mask_for((uint32_t)hashes[i+0]);
        __m256i m1 = mask_for((uint32_t)hashes[i+1]);
        __m256i m2 = mask_for((uint32_t)hashes[i+2]);
        __m256i m3 = mask_for((uint32_t)hashes[i+3]);
        __m256i c0 = _mm256_load_si256((__m256i*)p0);
        __m256i c1 = _mm256_load_si256((__m256i*)p1);
        __m256i c2 = _mm256_load_si256((__m256i*)p2);
        __m256i c3 = _mm256_load_si256((__m256i*)p3);
        hits += _mm256_testc_si256(c0, m0);
        hits += _mm256_testc_si256(c1, m1);
        hits += _mm256_testc_si256(c2, m2);
        hits += _mm256_testc_si256(c3, m3);
    }
    for (; i < n; i++) {
        if (QB_API(contains_prehash)(p, hashes[i])) hits++;
    }
    return hits;
}
