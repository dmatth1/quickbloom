// bench_atomic_insert.c -- compare non-atomic vs atomic insert paths
// for SBBF. Quantifies the cost of making concurrent insert safe.
//
// Non-atomic: the current path. One 256-bit aligned load + or + store.
// Atomic: split the 256-bit block as 4 x uint64 and do __atomic_fetch_or
// on each. Each atomic op compiles to a `lock or qword` on x86, which
// is the only x86 primitive that provides cross-core RMW atomicity at
// 64-bit width (cmpxchg16b is the only 128-bit option; no 256-bit RMW).
//
// Build: make bench-atomic
// Run:   ./build/bench_atomic_insert

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

// ====== minimal SBBF primitives copied from bloom_sbbf.c =================

typedef struct {
    uint32_t* bits;
    size_t    nblocks_mask;
} bloom_t;

static inline __m256i mask_for(uint32_t h32) {
    const __m256i salt = _mm256_set_epi32(
        (int)0x5c6bfb31u, (int)0x9efc4947u, (int)0x2df1424bu, (int)0x705495c7u,
        (int)0xa2b7289du, (int)0x8824ad5bu, (int)0x44974d91u, (int)0x47b6137bu);
    const __m256i hbcast = _mm256_set1_epi32((int)h32);
    const __m256i prod   = _mm256_mullo_epi32(hbcast, salt);
    const __m256i shift  = _mm256_srli_epi32(prod, 27);
    const __m256i ones   = _mm256_set1_epi32(1);
    return _mm256_sllv_epi32(ones, shift);
}

static inline uint32_t* block_for(const bloom_t* b, uint64_t h64) {
    size_t idx = ((size_t)(h64 >> 32)) & b->nblocks_mask;
    return b->bits + idx * 8;
}

// ====== current non-atomic insert (verbatim from bloom_sbbf.c) ============
__attribute__((noinline))
static void insert_nonatomic(bloom_t* b, uint64_t h) {
    uint32_t* blk = block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    __m256i cur = _mm256_load_si256((__m256i*)blk);
    _mm256_store_si256((__m256i*)blk, _mm256_or_si256(cur, m));
}

// ====== atomic insert: 4 x __atomic_fetch_or on the 4 uint64 words ========
__attribute__((noinline))
static void insert_atomic(bloom_t* b, uint64_t h) {
    uint64_t* blk = (uint64_t*)block_for(b, h);
    __m256i m = mask_for((uint32_t)h);
    uint64_t mask64[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i*)mask64, m);
    __atomic_fetch_or(&blk[0], mask64[0], __ATOMIC_RELAXED);
    __atomic_fetch_or(&blk[1], mask64[1], __ATOMIC_RELAXED);
    __atomic_fetch_or(&blk[2], mask64[2], __ATOMIC_RELAXED);
    __atomic_fetch_or(&blk[3], mask64[3], __ATOMIC_RELAXED);
}

// ====== bench harness =====================================================

#define N_BLOCKS   (1u << 17)        // 32 MB filter — out of L2, into L3
#define N_HASHES   (1u << 20)        // 1M pre-computed hashes (8 MB)
#define REPEATS    21
#define WARMUP     3

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

typedef void (*insert_fn)(bloom_t*, uint64_t);

static void measure(const char* name, insert_fn fn, bloom_t* b, const uint64_t* hashes) {
    uint64_t samples[REPEATS];
    // Warmup.
    for (int w = 0; w < WARMUP; w++) {
        for (size_t i = 0; i < N_HASHES; i++) fn(b, hashes[i]);
    }
    // Measure.
    for (int r = 0; r < REPEATS; r++) {
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < N_HASHES; i++) fn(b, hashes[i]);
        uint64_t t1 = now_ns();
        samples[r] = t1 - t0;
    }
    qsort(samples, REPEATS, sizeof(samples[0]), cmp_u64);
    double ns_min = (double)samples[0] / (double)N_HASHES;
    double ns_med = (double)samples[REPEATS / 2] / (double)N_HASHES;
    double ns_p90 = (double)samples[(REPEATS * 9) / 10] / (double)N_HASHES;
    printf("  %-22s  min=%6.2f  med=%6.2f  p90=%6.2f  ns/op\n",
           name, ns_min, ns_med, ns_p90);
}

int main(void) {
    bloom_t b;
    b.nblocks_mask = N_BLOCKS - 1;
    void* mem = NULL;
    if (posix_memalign(&mem, 32, (size_t)N_BLOCKS * 32) != 0) {
        perror("posix_memalign"); return 1;
    }
    memset(mem, 0, (size_t)N_BLOCKS * 32);
    b.bits = (uint32_t*)mem;

    uint64_t* hashes = (uint64_t*)aligned_alloc(64, N_HASHES * 8);
    if (!hashes) { perror("aligned_alloc"); return 1; }
    uint64_t s = 0xC0FFEEull;
    for (size_t i = 0; i < N_HASHES; i++) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        hashes[i] = z ^ (z >> 31);
    }

    printf("bench_atomic_insert: %u hashes into a %u-block filter (%.1f MB), %d repeats\n",
           N_HASHES, N_BLOCKS, (double)N_BLOCKS * 32.0 / (1024.0 * 1024.0), REPEATS);
    measure("insert (non-atomic)", insert_nonatomic, &b, hashes);
    measure("insert (atomic x4 u64)", insert_atomic, &b, hashes);

    free(hashes);
    free(mem);
    return 0;
}
