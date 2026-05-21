// test_quickbloom.c -- native correctness tests.
//
// What this checks (mirrors test_bloom.py):
//   1. Round-trip: insert N keys, query all back; zero false negatives.
//   2. False-positive ceiling: query N keys never inserted; FP rate is
//      below the sanity ceiling (5%).
//   3. Variable-length keys (1..64 bytes) exercise the fasthash64_var
//      tail path, not just the 16-byte fast path.
//   4. Prehash API: insert via qb_insert_prehash, contains via
//      qb_contains_prehash; round-trip with no false negatives.
//   5. qb_free(NULL) is a no-op.
//   6. qb_new(0) does not crash and returns a usable filter (one block).
//   7. qb_estimate_bits sanity: monotonic in n and 1/fp, never zero.

#include "quickbloom.h"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_INSERT 50000
#define N_QUERY  50000
#define FP_CEILING 0.05

static uint64_t pcg_state[2] = { 0xC0FFEEull, 0xDEADBEEFull };
static uint64_t pcg_next(void) {
    uint64_t x = pcg_state[0];
    uint64_t y = pcg_state[1];
    pcg_state[0] = y;
    x ^= x << 23;
    pcg_state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return pcg_state[1] + y;
}
static void pcg_seed(uint64_t s) {
    pcg_state[0] = s;
    pcg_state[1] = s ^ 0xDEADBEEFull;
}

static void fill_random(uint8_t* buf, size_t len) {
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t v = pcg_next();
        memcpy(buf + i, &v, 8);
        i += 8;
    }
    if (i < len) {
        uint64_t v = pcg_next();
        memcpy(buf + i, &v, len - i);
    }
}

static void mark_miss(uint8_t* buf) { buf[0] |= 0x80; }

#define ASSERT(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: " __VA_ARGS__); \
        fprintf(stderr, "\n  at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

// 1+2: round-trip and FP rate with random 16-byte keys.
static void test_round_trip_16(void) {
    void* f = qb_new(N_INSERT * 16);
    ASSERT(f != NULL, "qb_new returned NULL");

    uint8_t (*keys)[16] = malloc(N_INSERT * 16);
    ASSERT(keys != NULL, "alloc inserted keys");
    pcg_seed(0xC0FFEEull);
    for (int i = 0; i < N_INSERT; i++) {
        fill_random(keys[i], 16);
        keys[i][0] &= 0x7f;
        qb_insert(f, keys[i], 16);
    }
    for (int i = 0; i < N_INSERT; i++) {
        ASSERT(qb_contains(f, keys[i], 16),
               "false negative at i=%d", i);
    }

    int fps = 0;
    for (int i = 0; i < N_QUERY; i++) {
        uint8_t miss[16];
        fill_random(miss, 16);
        mark_miss(miss);
        if (qb_contains(f, miss, 16)) fps++;
    }
    double fp_rate = (double)fps / (double)N_QUERY;
    printf("  16B FP rate = %.4f (ceil %.2f)\n", fp_rate, FP_CEILING);
    ASSERT(fp_rate < FP_CEILING,
           "FP rate too high: %.4f >= %.2f", fp_rate, FP_CEILING);

    free(keys);
    qb_free(f);
}

// 3: variable-length keys (1..64 bytes).
static void test_round_trip_varlen(void) {
    void* f = qb_new(N_INSERT * 16);
    ASSERT(f != NULL, "qb_new returned NULL");

    enum { N = 2000 };
    uint8_t store[N][64];
    size_t  lens[N];
    pcg_seed(0xABCDEFull);
    for (int i = 0; i < N; i++) {
        size_t len = 1 + (i % 64);
        fill_random(store[i], len);
        store[i][0] &= 0x7f;
        lens[i] = len;
        qb_insert(f, store[i], len);
    }
    for (int i = 0; i < N; i++) {
        ASSERT(qb_contains(f, store[i], lens[i]),
               "varlen false negative at i=%d len=%zu", i, lens[i]);
    }
    qb_free(f);
}

// 4: prehash round-trip.
static void test_prehash(void) {
    void* f = qb_new(10000 * 16);
    ASSERT(f != NULL, "qb_new");

    uint64_t hashes[1000];
    pcg_seed(0xFACEFEEDull);
    for (int i = 0; i < 1000; i++) {
        hashes[i] = pcg_next();
        qb_insert_prehash(f, hashes[i]);
    }
    for (int i = 0; i < 1000; i++) {
        ASSERT(qb_contains_prehash(f, hashes[i]),
               "prehash false negative at i=%d", i);
    }
    qb_free(f);
}

// 5: qb_free(NULL) is a no-op.
static void test_free_null(void) {
    qb_free(NULL);
}

// 6: qb_new(0) returns a usable filter.
static void test_new_zero(void) {
    void* f = qb_new(0);
    ASSERT(f != NULL, "qb_new(0) returned NULL");
    const uint8_t key[16] = {0};
    qb_insert(f, key, 16);
    ASSERT(qb_contains(f, key, 16), "qb_new(0) contains after insert");
    qb_free(f);
}

// Structured (non-random) 16-byte inputs to probe the well-known
// weak spot of single-multiply hashes. README's "rejected variants"
// already documents one regression in this family (shift-based mask
// compute had weak diffusion in the low bits of the 128-bit product).
// This test is cheap insurance against future ones.
static void test_round_trip_structured(void) {
    void* f = qb_new(qb_estimate_bits(20000, 0.01));
    ASSERT(f != NULL, "qb_new");

    enum { N = 5000 };
    uint8_t keys[3 * N][16];

    // Pattern A: sequential u128 (little-endian counter in the low 8
    // bytes, zero in the high 8). Stresses correlated keys differing
    // by 1.
    for (int i = 0; i < N; i++) {
        memset(keys[i], 0, 16);
        uint64_t v = (uint64_t)i;
        memcpy(keys[i], &v, 8);
        qb_insert(f, keys[i], 16);
    }
    // Pattern B: ASCII-like — printable characters varying in the low
    // byte, fixed pad in the high 15. Tight clustering in hash input
    // space.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 16; j++) keys[N + i][j] = 0x41 + (j & 15);
        keys[N + i][0] = (uint8_t)((i % 26) + 0x41);
        keys[N + i][1] = (uint8_t)((i / 26) & 0x7f);
        qb_insert(f, keys[N + i], 16);
    }
    // Pattern C: UUIDv4-shaped — random bytes with version (0x4_) and
    // variant (0x8_/0x9_/0xa_/0xb_) bits in fixed positions. These
    // bits never vary across inputs, which is exactly the kind of
    // structure a weak hash mishandles.
    pcg_seed(0xC0DEC0DEull);
    for (int i = 0; i < N; i++) {
        fill_random(keys[2 * N + i], 16);
        keys[2 * N + i][6] = (uint8_t)(0x40 | (keys[2 * N + i][6] & 0x0f));
        keys[2 * N + i][8] = (uint8_t)(0x80 | (keys[2 * N + i][8] & 0x3f));
        qb_insert(f, keys[2 * N + i], 16);
    }

    // No false negatives across any pattern.
    for (int i = 0; i < 3 * N; i++) {
        ASSERT(qb_contains(f, keys[i], 16),
               "structured false negative at i=%d", i);
    }

    // FP rate must stay reasonable on adversarially-structured queries.
    // Build a query set with the same structural shape but disjoint
    // from the inserted set (offset the counter by a large constant).
    int fps = 0;
    for (int i = 0; i < N; i++) {
        uint8_t q[16] = {0};
        uint64_t v = (uint64_t)i + (1ULL << 40);  // far from inserted range
        memcpy(q, &v, 8);
        if (qb_contains(f, q, 16)) fps++;
    }
    double fp_rate = (double)fps / (double)N;
    printf("  structured sequential FP rate = %.4f (ceil %.2f)\n",
           fp_rate, FP_CEILING);
    ASSERT(fp_rate < FP_CEILING,
           "structured FP rate too high: %.4f", fp_rate);

    qb_free(f);
}

// 7: serialize/deserialize round-trip.
static void test_serialize_round_trip(void) {
    void* f = qb_new(N_INSERT * 16);
    ASSERT(f != NULL, "qb_new");

    pcg_seed(0xDEADBEEFull);
    enum { N = 5000 };
    uint8_t keys[N][16];
    for (int i = 0; i < N; i++) {
        fill_random(keys[i], 16);
        keys[i][0] &= 0x7f;
        qb_insert(f, keys[i], 16);
    }

    size_t nbytes = qb_serialized_size(f);
    ASSERT(nbytes > 0 && nbytes % 32 == 0, "serialized_size sanity");
    uint8_t* buf = malloc(nbytes);
    ASSERT(buf != NULL, "alloc serialize buf");
    qb_serialize(f, buf);

    void* g = qb_deserialize(buf, nbytes);
    ASSERT(g != NULL, "deserialize returned NULL");
    ASSERT(qb_serialized_size(g) == nbytes, "round-trip serialized_size");

    // Every inserted key must still be reported present in the
    // round-tripped filter.
    for (int i = 0; i < N; i++) {
        ASSERT(qb_contains(g, keys[i], 16),
               "round-trip false negative at i=%d", i);
    }

    // The byte image of g must match the byte image of f.
    uint8_t* buf2 = malloc(nbytes);
    qb_serialize(g, buf2);
    ASSERT(memcmp(buf, buf2, nbytes) == 0,
           "byte image differs after round trip");

    // Invalid inputs.
    ASSERT(qb_deserialize(NULL, 32) == NULL,        "deserialize(NULL) -> NULL");
    ASSERT(qb_deserialize(buf, 0) == NULL,          "deserialize(0 bytes) -> NULL");
    ASSERT(qb_deserialize(buf, 33) == NULL,         "deserialize(non-32-multiple) -> NULL");
    ASSERT(qb_deserialize(buf, 96) == NULL,         "deserialize(3 blocks, non-pow2) -> NULL");

    free(buf);
    free(buf2);
    qb_free(g);
    qb_free(f);
}

// 8: qb_estimate_bits sanity.
static void test_estimate_bits(void) {
    ASSERT(qb_estimate_bits(0,      0.01) >= 256, "estimate(0) >= 256");
    ASSERT(qb_estimate_bits(100,    0.01) >= 256, "estimate(small) >= 256");

    size_t m = qb_estimate_bits(1000000, 0.01);
    ASSERT(m >=  5 * 1000000UL, "1%% FP: at least 5 bits/key");
    ASSERT(m <= 20 * 1000000UL, "1%% FP: at most 20 bits/key");

    ASSERT(qb_estimate_bits(1000000, 0.001) > qb_estimate_bits(1000000, 0.01),
           "0.1%% FP needs more bits than 1%% FP");

    ASSERT(qb_estimate_bits(1000, 0.0)  > 0, "fp=0  doesn't crash");
    ASSERT(qb_estimate_bits(1000, 1.0)  > 0, "fp=1  doesn't crash");
    ASSERT(qb_estimate_bits(1000, -1.0) > 0, "negative fp doesn't crash");
}

int main(void) {
    printf("quickbloom test (version %s)\n", QUICKBLOOM_VERSION_STRING);
    printf("[ correctness ]\n");
    test_free_null();
    test_new_zero();
    test_round_trip_16();
    test_round_trip_varlen();
    test_round_trip_structured();
    test_prehash();
    printf("[ serialize round-trip ]\n");
    test_serialize_round_trip();
    printf("[ qb_estimate_bits ]\n");
    test_estimate_bits();
    printf("OK\n");
    return 0;
}
