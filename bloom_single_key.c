// bloom_single_key.c -- SBBF Bloom filter, single-key API, no prefetch.
//
// Recommended for: in-cache filters (S/M sizes; < ~10 MB), or when the
// filter size isn't known in advance and you want the lowest per-op
// in-cache latency.
//
// Algorithm: Split Block Bloom Filter (Apache Parquet spec) with:
//   - 256-bit blocks (8 x 32-bit words), K=8 (one bit per word)
//   - wymum hash (single 128-bit multiply) on 16-byte fast path,
//     fasthash64 fallback for variable-length keys
//   - Power-of-2 nblocks with bitmask block-index (no fastrange)
//   - AVX2 SIMD mask compute (vpmullo + vpsrli + vpsllv)
//   - 4-way unrolled bulk_contains
//   - bulk_insert uses sequential load+OR+store per key so hardware
//     store-to-load forwarding handles aliased blocks correctly
//
// Requirements: x86_64 with AVX2 + BMI2 (Intel Haswell 2013+, AMD
// Excavator 2015+/Zen 1+). Compile with -O3 -mavx2 -mbmi2 -mfma.
//
// On Intel Sapphire Rapids @ 2.1 GHz with clang -O3, prehash mode:
//   S (128 KB filter):  ~1.1 ns/op  (~2.4 cyc/op)
//   M (2 MB):           ~1.7 ns/op
//   L (32 MB):          ~5-6 ns/op
//   XL (512 MB):        ~17-19 ns/op (memory-bound, bigger wins via bloom_unified.c)
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

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

void* bloom_new(size_t nbits) {
    size_t nblocks = (nbits + 255) / 256;
    if (nblocks == 0) nblocks = 1;
    size_t pow2 = 1;
    while (pow2 < nblocks) pow2 <<= 1;
    nblocks = pow2;
    bloom_t* b = (bloom_t*)malloc(sizeof(bloom_t));
    b->nblocks_mask = nblocks - 1;
    size_t nbytes = nblocks * SBBF_BLOCK_BYTES;
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
    uint64_t h = bloom_hash(key, len);
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    _mm256_store_si256((__m256i*)blk, _mm256_or_si256(cur, m));
}

int bloom_contains(void* p, const void* key, size_t len) {
    bloom_t* b = (bloom_t*)p;
    uint64_t h = bloom_hash(key, len);
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    return _mm256_testc_si256(cur, m);
}

// 4-way unrolled bulk_insert: compute parallel, L+O+S sequential per key
// so hardware store-load forwarding handles aliased blocks correctly.
void bloom_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t i = 0;
    if (klen == 16) {
        for (; i + 4 <= n; i += 4) {
            const uint8_t* k0 = keys + (i+0)*16;
            const uint8_t* k1 = keys + (i+1)*16;
            const uint8_t* k2 = keys + (i+2)*16;
            const uint8_t* k3 = keys + (i+3)*16;
            // Compute -- all parallel
            uint64_t h0 = hash16(k0);
            uint64_t h1 = hash16(k1);
            uint64_t h2 = hash16(k2);
            uint64_t h3 = hash16(k3);
            uint32_t* p0 = block_for(b, h0);
            uint32_t* p1 = block_for(b, h1);
            uint32_t* p2 = block_for(b, h2);
            uint32_t* p3 = block_for(b, h3);
            __m256i m0 = mask_for((uint32_t)h0);
            __m256i m1 = mask_for((uint32_t)h1);
            __m256i m2 = mask_for((uint32_t)h2);
            __m256i m3 = mask_for((uint32_t)h3);
            // L+O+S serialized per key (within iter). Correctness: if any
            // pair of pointers aliases, the second load picks up the
            // first store via store-load forwarding in the load buffer.
            {   __m256i c = _mm256_load_si256((__m256i*)p0);
                _mm256_store_si256((__m256i*)p0, _mm256_or_si256(c, m0)); }
            {   __m256i c = _mm256_load_si256((__m256i*)p1);
                _mm256_store_si256((__m256i*)p1, _mm256_or_si256(c, m1)); }
            {   __m256i c = _mm256_load_si256((__m256i*)p2);
                _mm256_store_si256((__m256i*)p2, _mm256_or_si256(c, m2)); }
            {   __m256i c = _mm256_load_si256((__m256i*)p3);
                _mm256_store_si256((__m256i*)p3, _mm256_or_si256(c, m3)); }
        }
    }
    for (; i < n; i++) bloom_insert(b, keys + i * klen, klen);
}

size_t bloom_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t hits = 0;
    size_t i = 0;
    if (klen == 16) {
        for (; i + 4 <= n; i += 4) {
            const uint8_t* k0 = keys + (i+0)*16;
            const uint8_t* k1 = keys + (i+1)*16;
            const uint8_t* k2 = keys + (i+2)*16;
            const uint8_t* k3 = keys + (i+3)*16;
            uint64_t h0 = hash16(k0);
            uint64_t h1 = hash16(k1);
            uint64_t h2 = hash16(k2);
            uint64_t h3 = hash16(k3);
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
        if (bloom_contains(b, keys + i * klen, klen)) hits++;
    }
    return hits;
}

// Prehashed ABI
void bloom_insert_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    _mm256_store_si256((__m256i*)blk, _mm256_or_si256(cur, m));
}

int bloom_contains_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    return _mm256_testc_si256(cur, m);
}

void bloom_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint64_t h0 = hashes[i+0], h1 = hashes[i+1];
        uint64_t h2 = hashes[i+2], h3 = hashes[i+3];
        uint32_t* p0 = block_for(b, h0);
        uint32_t* p1 = block_for(b, h1);
        uint32_t* p2 = block_for(b, h2);
        uint32_t* p3 = block_for(b, h3);
        __m256i m0 = mask_for((uint32_t)h0);
        __m256i m1 = mask_for((uint32_t)h1);
        __m256i m2 = mask_for((uint32_t)h2);
        __m256i m3 = mask_for((uint32_t)h3);
        {   __m256i c = _mm256_load_si256((__m256i*)p0);
            _mm256_store_si256((__m256i*)p0, _mm256_or_si256(c, m0)); }
        {   __m256i c = _mm256_load_si256((__m256i*)p1);
            _mm256_store_si256((__m256i*)p1, _mm256_or_si256(c, m1)); }
        {   __m256i c = _mm256_load_si256((__m256i*)p2);
            _mm256_store_si256((__m256i*)p2, _mm256_or_si256(c, m2)); }
        {   __m256i c = _mm256_load_si256((__m256i*)p3);
            _mm256_store_si256((__m256i*)p3, _mm256_or_si256(c, m3)); }
    }
    for (; i < n; i++) bloom_insert_prehash(p, hashes[i]);
}

size_t bloom_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    bloom_t* b = (bloom_t*)p;
    size_t hits = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint64_t h0 = hashes[i+0], h1 = hashes[i+1];
        uint64_t h2 = hashes[i+2], h3 = hashes[i+3];
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
    for (; i < n; i++) {
        if (bloom_contains_prehash(b, hashes[i])) hits++;
    }
    return hits;
}
