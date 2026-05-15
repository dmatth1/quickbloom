// bloom_krassovsky.c -- Krassovsky's PatternedSimdBloomFilter, faithful port.
//
// Reference: github.com/save-buffer/bloomfilter_benchmarks (MIT). This is
// the design published as "2.5 cycles/op" by Save Buffer, and the entry
// the README's comparison table references as "Krassovsky PatternedSimd".
//
// Algorithm differs structurally from SBBF:
//   - 64-bit "blocks" (one uint64_t per block), not 256-bit.
//     num_blocks is a power of 2.
//   - 1024-entry pre-computed mask table, each entry is a 57-bit pattern
//     with 4-5 bits set. Sliding-window construction so adjacent indices
//     give different but related masks; one mask per insert/query is
//     picked via the low 10 hash bits.
//   - 6 hash bits choose how much to ROTATE the chosen mask before use.
//   - Remaining hash bits index into the bit vector to pick a block.
//   - Batched 8-way: AVX2 processes 4 keys at a time (one __m256i =
//     4 x uint64_t hashes); two such regs per batch8 call.
//   - Key win on contains: _mm256_i64gather_epi64 loads 4 blocks per
//     instruction. With 64-bit blocks, gather amortizes nicely.
//
// Used as a real external competitor: the batched paths here let
// bench_all.py compare the gather-based design against this repo's
// `bloom_batched.c` (which deliberately avoids gather).
//
// Adaptations for this repo: wymum bloom_hash so the hash+bloom path is
// directly comparable; the rest of the algorithm (mask table, AVX2
// sequence, ConstructMask, GetBlockIdx, gather) is faithful to Save
// Buffer's source.

#define K_HASHES 5

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

// --- Mask table -----------------------------------------------------------
#define LOG_NUM_MASKS 10
#define NUM_MASKS    (1 << LOG_NUM_MASKS)
#define BITS_PER_MASK 57
#define MIN_BITS_SET  4
#define MAX_BITS_SET  5
#define ROTATE_BITS   6
#define MASK_BYTES   ((NUM_MASKS + 64) / 8)

static uint8_t MASK_TABLE[MASK_BYTES];
static int mask_table_inited = 0;

static uint64_t prng_state = 0;
static inline uint32_t prng_next(void) {
    prng_state ^= prng_state >> 12;
    prng_state ^= prng_state << 25;
    prng_state ^= prng_state >> 27;
    return (uint32_t)(prng_state * 0x2545f4914f6cdd1dULL >> 32);
}

static void mask_table_init(void) {
    if (mask_table_inited) return;
    memset(MASK_TABLE, 0, sizeof(MASK_TABLE));
    prng_state = 0x123456789abcdef0ULL;

    int range = MAX_BITS_SET - MIN_BITS_SET + 1;
    int num_set = MIN_BITS_SET + (int)(prng_next() % (uint32_t)range);
    for (int i = 0; i < num_set; i++) {
        int pos;
        do {
            pos = (int)(prng_next() % BITS_PER_MASK);
        } while ((MASK_TABLE[pos / 8] >> (pos % 8)) & 1);
        MASK_TABLE[pos / 8] |= (uint8_t)(1u << (pos % 8));
    }

    int total_bits = NUM_MASKS + BITS_PER_MASK - 1;
    int cur = num_set;
    for (int i = BITS_PER_MASK; i < total_bits; i++) {
        int leaving_idx = i - BITS_PER_MASK;
        int leaving = (MASK_TABLE[leaving_idx / 8] >> (leaving_idx % 8)) & 1;
        if (leaving == 1 && cur == MIN_BITS_SET) {
            MASK_TABLE[i / 8] |= (uint8_t)(1u << (i % 8));
            continue;
        }
        if (leaving == 0 && cur == MAX_BITS_SET) continue;
        if ((int)(prng_next() % (BITS_PER_MASK * 2)) < MIN_BITS_SET + MAX_BITS_SET) {
            MASK_TABLE[i / 8] |= (uint8_t)(1u << (i % 8));
            if (leaving == 0) cur += 1;
        } else {
            if (leaving == 1) cur -= 1;
        }
    }
    mask_table_inited = 1;
}

// --- Bloom struct ---------------------------------------------------------
typedef struct {
    uint64_t* bv;
    uint64_t  num_blocks;
    uint64_t  num_blocks_mask;
} bloom_t;

void* bloom_new(size_t nbits) {
    mask_table_init();
    bloom_t* b = (bloom_t*)malloc(sizeof(bloom_t));
    // Round nbits up to next pow2; num_blocks = nbits / 64.
    size_t pow2_bits = 64;
    while (pow2_bits < nbits) pow2_bits <<= 1;
    b->num_blocks = pow2_bits / 64;
    if (b->num_blocks == 0) b->num_blocks = 1;
    b->num_blocks_mask = b->num_blocks - 1;
    size_t nbytes = b->num_blocks * sizeof(uint64_t);
    void* mem = NULL;
    if (posix_memalign(&mem, 32, nbytes) != 0) mem = calloc(nbytes, 1);
    else memset(mem, 0, nbytes);
    b->bv = (uint64_t*)mem;
    return b;
}

void bloom_free(void* p) {
    bloom_t* b = (bloom_t*)p;
    free(b->bv);
    free(b);
}

// --- Hash (for the hash+bloom path; matches v14) -------------------------
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

// --- Scalar single-key path ----------------------------------------------
static inline uint64_t construct_mask_scalar(uint64_t h) {
    uint32_t idx = (uint32_t)(h & ((1u << LOG_NUM_MASKS) - 1));
    uint64_t raw;
    // MASK_TABLE has at least MASK_BYTES = 136 bytes; idx/8 <= 127, so
    // reading 8 bytes at idx/8 is safe.
    memcpy(&raw, MASK_TABLE + (idx >> 3), 8);
    uint64_t unrot = (raw >> (idx & 7)) & ((1ULL << BITS_PER_MASK) - 1);
    uint32_t rot = (uint32_t)((h >> LOG_NUM_MASKS) & ((1u << ROTATE_BITS) - 1));
    uint64_t up = unrot << rot;
    uint64_t down = (rot == 0) ? 0 : (unrot >> (64 - rot));
    return up | down;
}

static inline uint64_t block_idx_scalar(const bloom_t* b, uint64_t h) {
    return (h >> (LOG_NUM_MASKS + ROTATE_BITS)) & b->num_blocks_mask;
}

void bloom_insert_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    b->bv[block_idx_scalar(b, h)] |= construct_mask_scalar(h);
}

int bloom_contains_prehash(void* p, uint64_t h) {
    bloom_t* b = (bloom_t*)p;
    uint64_t mask = construct_mask_scalar(h);
    return (b->bv[block_idx_scalar(b, h)] & mask) == mask;
}

void bloom_insert(void* p, const void* key, size_t len) {
    bloom_insert_prehash(p, bloom_hash(key, len));
}

int bloom_contains(void* p, const void* key, size_t len) {
    return bloom_contains_prehash(p, bloom_hash(key, len));
}

// --- AVX2 mask construction (4 hashes per __m256i) -----------------------
static inline __m256i construct_mask_avx2(__m256i hash) {
    const __m256i mask_idx_mask = _mm256_set1_epi64x((1LL << LOG_NUM_MASKS) - 1);
    const __m256i mask_57       = _mm256_set1_epi64x((1LL << BITS_PER_MASK) - 1);
    const __m256i v64           = _mm256_set1_epi64x(64);
    const __m256i rot_mask      = _mm256_set1_epi64x((1LL << ROTATE_BITS) - 1);
    const __m256i seven         = _mm256_set1_epi64x(0x7);

    __m256i idx = _mm256_and_si256(hash, mask_idx_mask);
    __m256i byte_idx = _mm256_srli_epi64(idx, 3);
    __m256i bit_idx  = _mm256_and_si256(idx, seven);
    __m256i raw = _mm256_i64gather_epi64((const long long*)MASK_TABLE, byte_idx, 1);
    __m256i unrot = _mm256_and_si256(_mm256_srlv_epi64(raw, bit_idx), mask_57);

    __m256i rot = _mm256_and_si256(_mm256_srli_epi64(hash, LOG_NUM_MASKS), rot_mask);
    __m256i up   = _mm256_sllv_epi64(unrot, rot);
    __m256i down = _mm256_srlv_epi64(unrot, _mm256_sub_epi64(v64, rot));
    return _mm256_or_si256(up, down);
}

static inline __m256i block_idx_avx2(const bloom_t* b, __m256i hash) {
    const __m256i nbm = _mm256_set1_epi64x((int64_t)b->num_blocks_mask);
    __m256i idx = _mm256_srli_epi64(hash, LOG_NUM_MASKS + ROTATE_BITS);
    return _mm256_and_si256(idx, nbm);
}

// --- Batched 8-way ABI ---------------------------------------------------
void bloom_insert_batch8(void* p, const uint64_t* h) {
    bloom_t* b = (bloom_t*)p;
    __m256i hA = _mm256_loadu_si256((const __m256i*)(h + 0));
    __m256i hB = _mm256_loadu_si256((const __m256i*)(h + 4));
    __m256i mA = construct_mask_avx2(hA);
    __m256i mB = construct_mask_avx2(hB);
    __m256i iA = block_idx_avx2(b, hA);
    __m256i iB = block_idx_avx2(b, hB);
    // AVX2 has no integer scatter; extract + scalar OR.
    b->bv[(uint64_t)_mm256_extract_epi64(iA, 0)] |= (uint64_t)_mm256_extract_epi64(mA, 0);
    b->bv[(uint64_t)_mm256_extract_epi64(iA, 1)] |= (uint64_t)_mm256_extract_epi64(mA, 1);
    b->bv[(uint64_t)_mm256_extract_epi64(iA, 2)] |= (uint64_t)_mm256_extract_epi64(mA, 2);
    b->bv[(uint64_t)_mm256_extract_epi64(iA, 3)] |= (uint64_t)_mm256_extract_epi64(mA, 3);
    b->bv[(uint64_t)_mm256_extract_epi64(iB, 0)] |= (uint64_t)_mm256_extract_epi64(mB, 0);
    b->bv[(uint64_t)_mm256_extract_epi64(iB, 1)] |= (uint64_t)_mm256_extract_epi64(mB, 1);
    b->bv[(uint64_t)_mm256_extract_epi64(iB, 2)] |= (uint64_t)_mm256_extract_epi64(mB, 2);
    b->bv[(uint64_t)_mm256_extract_epi64(iB, 3)] |= (uint64_t)_mm256_extract_epi64(mB, 3);
}

uint8_t bloom_contains_batch8(void* p, const uint64_t* h) {
    bloom_t* b = (bloom_t*)p;
    __m256i hA = _mm256_loadu_si256((const __m256i*)(h + 0));
    __m256i hB = _mm256_loadu_si256((const __m256i*)(h + 4));
    __m256i mA = construct_mask_avx2(hA);
    __m256i mB = construct_mask_avx2(hB);
    __m256i iA = block_idx_avx2(b, hA);
    __m256i iB = block_idx_avx2(b, hB);
    // The 64-bit-block gather is the structural win of this algorithm.
    __m256i bA = _mm256_i64gather_epi64((const long long*)b->bv, iA, 8);
    __m256i bB = _mm256_i64gather_epi64((const long long*)b->bv, iB, 8);
    __m256i cA = _mm256_cmpeq_epi64(_mm256_and_si256(mA, bA), mA);
    __m256i cB = _mm256_cmpeq_epi64(_mm256_and_si256(mB, bB), mB);
    // movemask_epi8 sets one bit per byte; a 64-bit lane that passed
    // cmpeq has all 8 bytes = 0xFF, so the lane's "did it match" bit
    // is the bit at position {0,8,16,24} of movemask output.
    int32_t mAm = _mm256_movemask_epi8(cA);
    int32_t mBm = _mm256_movemask_epi8(cB);
    uint8_t r = 0;
    r |= (uint8_t)(((mAm >>  0) & 1) << 0);
    r |= (uint8_t)(((mAm >>  8) & 1) << 1);
    r |= (uint8_t)(((mAm >> 16) & 1) << 2);
    r |= (uint8_t)(((mAm >> 24) & 1) << 3);
    r |= (uint8_t)(((mBm >>  0) & 1) << 4);
    r |= (uint8_t)(((mBm >>  8) & 1) << 5);
    r |= (uint8_t)(((mBm >> 16) & 1) << 6);
    r |= (uint8_t)(((mBm >> 24) & 1) << 7);
    return r;
}

void bloom_insert_batch8_bulk(void* p, const uint64_t* hashes, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) bloom_insert_batch8(p, hashes + i);
    for (; i < n; i++) bloom_insert_prehash(p, hashes[i]);
}

size_t bloom_contains_batch8_bulk(void* p, const uint64_t* hashes, size_t n) {
    size_t hits = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) hits += __builtin_popcount(bloom_contains_batch8(p, hashes + i));
    for (; i < n; i++) {
        if (bloom_contains_prehash(p, hashes[i])) hits++;
    }
    return hits;
}

// --- Key-based bulk: hash inline, then call batch8 ----------------------
void bloom_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    size_t i = 0;
    if (klen == 16 && n >= 8) {
        for (; i + 8 <= n; i += 8) {
            uint64_t h[8];
            for (int j = 0; j < 8; j++) h[j] = hash16(keys + (i+j) * 16);
            bloom_insert_batch8(p, h);
        }
    }
    for (; i < n; i++) bloom_insert(p, keys + i * klen, klen);
}

size_t bloom_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    size_t hits = 0;
    size_t i = 0;
    if (klen == 16 && n >= 8) {
        for (; i + 8 <= n; i += 8) {
            uint64_t h[8];
            for (int j = 0; j < 8; j++) h[j] = hash16(keys + (i+j) * 16);
            hits += __builtin_popcount(bloom_contains_batch8(p, h));
        }
    }
    for (; i < n; i++) {
        if (bloom_contains(p, keys + i * klen, klen)) hits++;
    }
    return hits;
}

// --- Prehash bulk (forwards to batch8 since the algorithm is gather-friendly)
void bloom_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    bloom_insert_batch8_bulk(p, hashes, n);
}

size_t bloom_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    return bloom_contains_batch8_bulk(p, hashes, n);
}
