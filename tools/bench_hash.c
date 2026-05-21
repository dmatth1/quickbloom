// bench_hash.c -- per-key cost of the three hash functions referenced in
// the README's perf section, measured on this host with the same
// compiler flags the rest of the bench uses. Lets a reader reconstruct
// hash+bloom latency from the prehash table by adding the hash cost.
//
//   wymum16     -- quickbloom's 16-byte fast path: one 128-bit multiply
//                  + xor-fold. Same code path used by the SBBF variants.
//   XXH64       -- Apache Parquet's mandated hash; used by impala and
//                  by arrow_rs (Rust parquet crate).
//   SipHash-1-3 -- fastbloom's default Hasher; 1 compression round, 3
//                  finalization rounds. Variant of standard SipHash-2-4.
//
// Build: make bench-hash    (or by hand:
//          cc -O3 -mavx2 -mbmi2 -mfma -maes -std=c11 -Wall -Wextra
//             tools/bench_hash.c -o build/bench_hash)
//
// XXH64 vendored from the xxHash reference (Yann Collet, BSD-2).
// SipHash-1-3 vendored from veorq/SipHash (CC0 / public domain).

#define _POSIX_C_SOURCE 200809L  // for clock_gettime under -std=c11

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =============================================================================
// wymum16 -- 16-byte fast path used by quickbloom (see bloom_sbbf.c).
// =============================================================================
static inline uint64_t wymum16(const void* data) {
    uint64_t a, b;
    memcpy(&a, data, 8);
    memcpy(&b, (const uint8_t*)data + 8, 8);
    // Same seeded XOR as quickbloom.c hash16 — prevents zero-input
    // collapse on structured keys with a zero half.
    a ^= 0x736f6d6570736575ULL;
    b ^= 0x646f72616e646f6dULL;
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
}

// XXH64 vendored from the canonical reference (BSD-2). One copy
// lives in tools/xxh64.h, included here and by tools/xxh64_helper.c.
#include "xxh64.h"

// =============================================================================
// SipHash-1-3 -- 1 compression round per message block, 3 finalization rounds.
// Variant of standard SipHash-2-4 (c=2,d=4). Same shape as Rust's default
// Hasher; fastbloom uses this with a per-instance random 128-bit key.
// =============================================================================
#define SIP_ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))
#define SIPROUND \
    do { \
        v0 += v1; v1 = SIP_ROTL(v1, 13); v1 ^= v0; v0 = SIP_ROTL(v0, 32); \
        v2 += v3; v3 = SIP_ROTL(v3, 16); v3 ^= v2; \
        v0 += v3; v3 = SIP_ROTL(v3, 21); v3 ^= v0; \
        v2 += v1; v1 = SIP_ROTL(v1, 17); v1 ^= v2; v2 = SIP_ROTL(v2, 32); \
    } while (0)

static uint64_t siphash13(const uint8_t* in, size_t inlen, const uint8_t k[16]) {
    uint64_t v0 = 0x736f6d6570736575ULL;
    uint64_t v1 = 0x646f72616e646f6dULL;
    uint64_t v2 = 0x6c7967656e657261ULL;
    uint64_t v3 = 0x7465646279746573ULL;
    uint64_t k0, k1, m;
    memcpy(&k0, k, 8);
    memcpy(&k1, k + 8, 8);
    v3 ^= k1; v2 ^= k0; v1 ^= k1; v0 ^= k0;

    const uint8_t* end = in + (inlen - (inlen % 8));
    const int left = (int)(inlen & 7);
    uint64_t b = ((uint64_t)inlen) << 56;
    for (; in != end; in += 8) {
        memcpy(&m, in, 8);
        v3 ^= m;
        SIPROUND;  // c=1 compression round
        v0 ^= m;
    }
    switch (left) {
        case 7: b |= ((uint64_t)in[6]) << 48; __attribute__((fallthrough));
        case 6: b |= ((uint64_t)in[5]) << 40; __attribute__((fallthrough));
        case 5: b |= ((uint64_t)in[4]) << 32; __attribute__((fallthrough));
        case 4: b |= ((uint64_t)in[3]) << 24; __attribute__((fallthrough));
        case 3: b |= ((uint64_t)in[2]) << 16; __attribute__((fallthrough));
        case 2: b |= ((uint64_t)in[1]) << 8;  __attribute__((fallthrough));
        case 1: b |= ((uint64_t)in[0]);
        case 0: break;
    }
    v3 ^= b;
    SIPROUND;  // c=1 compression round
    v0 ^= b;
    v2 ^= 0xff;
    SIPROUND; SIPROUND; SIPROUND;  // d=3 finalization rounds
    return v0 ^ v1 ^ v2 ^ v3;
}

// =============================================================================
// Bench harness.
// =============================================================================
#define KLEN     16
#define N_KEYS   200000     // matches the bench harness's S/M/L pool sizes
#define REPEATS  31         // odd so median is well-defined
#define WARMUP   3

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

// Pre-generate N deterministic 16-byte keys. Each key is the SHA-256
// of (seed, i) truncated to 16 bytes; cheap-but-good for bench inputs.
// Actually we just use a wymum-like PRNG to fill them; we don't care
// about cryptographic quality.
static void fill_keys(uint8_t* dst, size_t n_keys) {
    uint64_t s = 0xC0FFEEull;
    for (size_t i = 0; i < n_keys * KLEN; i += 8) {
        // splitmix64 step
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        memcpy(dst + i, &z, 8);
    }
}

typedef uint64_t (*hash_fn)(const void* key);

// Wrapper signatures so all three hashes share a (void*) -> u64 shape.
static uint64_t bench_wymum(const void* k)  { return wymum16(k); }
static uint64_t bench_xxh64(const void* k)  { return qb_xxh64(k, KLEN, 0); }
static const uint8_t SIP_KEY[16] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f
};
static uint64_t bench_siphash13(const void* k) {
    return siphash13((const uint8_t*)k, KLEN, SIP_KEY);
}

// Time `fn` over the full key pool for `iters` iterations, return median ns/op.
static void measure(const char* name, hash_fn fn, const uint8_t* keys) {
    uint64_t samples[REPEATS];
    uint64_t sink = 0;

    // Warmup.
    for (int w = 0; w < WARMUP; w++) {
        for (size_t i = 0; i < N_KEYS; i++) sink ^= fn(keys + i * KLEN);
    }
    // Measure.
    for (int r = 0; r < REPEATS; r++) {
        uint64_t t0 = now_ns();
        for (size_t i = 0; i < N_KEYS; i++) sink ^= fn(keys + i * KLEN);
        uint64_t t1 = now_ns();
        samples[r] = t1 - t0;
    }
    // Make sure the compiler can't drop the loop.
    if (sink == 0xDEADBEEFDEADBEEFULL) printf("(sink hit; impossible)\n");

    qsort(samples, REPEATS, sizeof(samples[0]), cmp_u64);
    double ns_min = (double)samples[0] / (double)N_KEYS;
    double ns_med = (double)samples[REPEATS / 2] / (double)N_KEYS;
    double ns_p90 = (double)samples[(REPEATS * 9) / 10] / (double)N_KEYS;
    printf("  %-14s  min=%5.2f  med=%5.2f  p90=%5.2f  ns/op\n",
           name, ns_min, ns_med, ns_p90);
}

int main(void) {
    uint8_t* keys = (uint8_t*)aligned_alloc(64, N_KEYS * KLEN);
    if (!keys) { perror("aligned_alloc"); return 1; }
    fill_keys(keys, N_KEYS);

    printf("bench_hash: %d 16-byte keys, %d repeats (warmup=%d)\n",
           N_KEYS, REPEATS, WARMUP);
    measure("wymum16",     bench_wymum,    keys);
    measure("xxh64",       bench_xxh64,    keys);
    measure("siphash-1-3", bench_siphash13, keys);

    free(keys);
    return 0;
}
