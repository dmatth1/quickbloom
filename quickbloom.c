// quickbloom.c -- the quickbloom SBBF implementation.
//
// One file, one ABI: the fastest single-key SBBF probe kernel on AVX2
// x86_64 across all three cache regimes we benchmark. See the README's
// Performance section for the comparison against other Bloom designs.
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

// K_HASHES 8 -- documented for bench tooling that scans for this.
#define K_HASHES 8

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

static inline uint64_t hash16(const void* data) {
    uint64_t a, b;
    memcpy(&a, data, 8);
    memcpy(&b, (const uint8_t*)data + 8, 8);
    // XOR with mixing constants so structured inputs (notably, a key
    // with a zero half — e.g. a 64-bit ID padded into a 16-byte
    // buffer) don't degenerate to multiplication by zero. Constants
    // are SipHash's initial state words; any pair of high-entropy
    // primes works.
    a ^= 0x736f6d6570736575ULL;
    b ^= 0x646f72616e646f6dULL;
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
    size_t   nblocks_mask;   // nblocks - 1 (nblocks is a power of two)
    uint32_t idx_shift;      // 32 - log2(nblocks); see block_for()
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
    // Parquet spec: idx = ((h >> 32) * num_blocks) >> 32  (fastrange).
    // We require num_blocks to be a power of two, so the multiply-and-
    // shift collapses to a single right shift of the upper 32-bit half
    // of the hash. Cast through uint64_t so an idx_shift of 32 (which
    // occurs when nblocks == 1) is well-defined and yields 0 instead
    // of being UB.
    uint32_t h32 = (uint32_t)(h64 >> 32);
    size_t idx = (size_t)(((uint64_t)h32) >> b->idx_shift);
    return b->bits + idx * 8;
}

// Helper: log2(n) for n a power of two, used to compute idx_shift.
static inline uint32_t log2_pow2(size_t n) {
    uint32_t k = 0;
    while ((((size_t)1) << k) < n) k++;
    return k;
}

// qb_new returns NULL on allocation failure. The returned pointer
// must be released with qb_free.
void* qb_new(size_t nbits) {
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
    b->idx_shift = 32 - log2_pow2(nblocks);
    b->bits = (uint32_t*)mem;
    return b;
}

void qb_free(void* p) {
    if (!p) return;
    bloom_t* b = (bloom_t*)p;
    free(b->bits);
    free(b);
}

void qb_insert(void* p, const void* key, size_t len) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h = bloom_hash(key, len);
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    _mm256_store_si256((__m256i*)blk, _mm256_or_si256(cur, m));
}

int qb_contains(void* p, const void* key, size_t len) {
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

void qb_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t i = 0;
    if (klen == 16) {
        for (; i + 4 <= n; i += 4) {
            uint64_t h0 = hash16(keys + (i+0)*16);
            uint64_t h1 = hash16(keys + (i+1)*16);
            uint64_t h2 = hash16(keys + (i+2)*16);
            uint64_t h3 = hash16(keys + (i+3)*16);
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
    for (; i < n; i++) qb_insert(b, keys + i * klen, klen);
}

size_t qb_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t hits = 0;
    size_t i = 0;
    if (klen == 16) {
        for (; i + 4 <= n; i += 4) {
            uint64_t h0 = hash16(keys + (i+0)*16);
            uint64_t h1 = hash16(keys + (i+1)*16);
            uint64_t h2 = hash16(keys + (i+2)*16);
            uint64_t h3 = hash16(keys + (i+3)*16);
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
        if (qb_contains(b, keys + i * klen, klen)) hits++;
    }
    return hits;
}

void qb_insert_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    _mm256_store_si256((__m256i*)blk, _mm256_or_si256(cur, m));
}

int qb_contains_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    return _mm256_testc_si256(cur, m);
}

void qb_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
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
    for (; i < n; i++) qb_insert_prehash(p, hashes[i]);
}

size_t qb_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t hits = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
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
        if (qb_contains_prehash(p, hashes[i])) hits++;
    }
    return hits;
}

// ---------------------------------------------------------------
// Serialization. The on-disk layout is identical to the in-memory
// one (nblocks 32-byte blocks of 8 little-endian uint32 lanes), which
// is also the Parquet spec layout. memcpy on both directions; on
// non-x86 the load/store macros would need byteswapping, but we're
// x86-only by build.

// Maximum nbytes qb_deserialize will accept. Larger inputs are
// rejected with NULL so an attacker-controlled Parquet bloom header
// can't trigger a multi-gigabyte posix_memalign. Override at compile
// time with -DQB_DESERIALIZE_MAX_BYTES=... if you need a larger
// ceiling. Default 1 GiB.
#ifndef QB_DESERIALIZE_MAX_BYTES
#define QB_DESERIALIZE_MAX_BYTES ((size_t)1 << 30)
#endif

size_t qb_serialized_size(void* p) {
    assert(p != NULL && "qb_serialized_size: filter pointer must be non-NULL");
    if (!p) return 0;
    bloom_t* b = (bloom_t*)p;
    return (b->nblocks_mask + 1) * SBBF_BLOCK_BYTES;
}

void qb_serialize(void* p, uint8_t* dst) {
    assert(p   != NULL && "qb_serialize: filter pointer must be non-NULL");
    assert(dst != NULL && "qb_serialize: dst buffer must be non-NULL");
    if (!p || !dst) return;
    bloom_t* b = (bloom_t*)p;
    size_t nbytes = (b->nblocks_mask + 1) * SBBF_BLOCK_BYTES;
    memcpy(dst, b->bits, nbytes);
}

void* qb_deserialize(const uint8_t* bytes, size_t nbytes) {
    if (!bytes) return NULL;
    if (nbytes == 0 || nbytes > QB_DESERIALIZE_MAX_BYTES) return NULL;
    if (nbytes % SBBF_BLOCK_BYTES != 0) return NULL;
    size_t nblocks = nbytes / SBBF_BLOCK_BYTES;
    // nblocks must be a power of two so the fastrange index reduces
    // to a shift (matches the qb_new layout).
    if ((nblocks & (nblocks - 1)) != 0) return NULL;

    bloom_t* b = (bloom_t*)malloc(sizeof(bloom_t));
    if (!b) return NULL;
    void* mem = NULL;
    if (posix_memalign(&mem, 32, nbytes) != 0) {
        free(b);
        return NULL;
    }
    memcpy(mem, bytes, nbytes);
    b->nblocks_mask = nblocks - 1;
    b->idx_shift = 32 - log2_pow2(nblocks);
    b->bits = (uint32_t*)mem;
    return b;
}
