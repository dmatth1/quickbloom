// bench_qb.c -- native C bench for the quickbloom kernel.
//
// Same shape as bench_hash.c, but for the bloom probe path. Lives
// next to the library so timing is inside C (clock_gettime, no FFI
// overhead) and the headline ns/op numbers don't carry the ctypes
// dispatch the Python harness adds.
//
// Reports min/med/p90 ns/op for:
//   insert         hash+bloom qb_insert
//   miss / hit     hash+bloom qb_contains on unseen / inserted keys
//   prehash *      same three on the qb_*_prehash entry points
// at three filter sizes spanning the cache hierarchy: S in-L2,
// M past per-core L2, L around L3.
//
// Build: make bench-qb. Sizes are hardcoded to match what
// harness.py auto-detects on a 1 MB-L2 / 33 MB-L3 host (the README
// reference). Override with QB_NBITS / QB_N_INSERT env vars to point
// at a different filter footprint.

#define _POSIX_C_SOURCE 200809L

#include "quickbloom.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define KLEN     16
#define REPEATS  11
#define WARMUP   2
#define ROUND64(n) (((n) + 63) & ~(size_t)63)

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

// splitmix64 — same generator bench_hash.c uses, so the two binaries
// stay in sync on input distribution.
static uint64_t sm_next(uint64_t* s) {
    *s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = (*s ^ (*s >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void fill_bytes(uint8_t* dst, size_t n_bytes, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n_bytes; i += 8) {
        uint64_t v = sm_next(&s);
        size_t take = (n_bytes - i < 8) ? (n_bytes - i) : 8;
        memcpy(dst + i, &v, take);
    }
}

static void fill_hashes(uint64_t* dst, size_t n, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; i++) dst[i] = sm_next(&s);
}

typedef struct { double min, med, p90; } stats_t;

static stats_t summarize(uint64_t* samples, int reps, size_t n_ops) {
    qsort(samples, reps, sizeof(samples[0]), cmp_u64);
    return (stats_t){
        .min = (double)samples[0]              / (double)n_ops,
        .med = (double)samples[reps / 2]       / (double)n_ops,
        .p90 = (double)samples[(reps * 9) / 10] / (double)n_ops,
    };
}

static void print_row(const char* label, stats_t s) {
    printf("  %-22s  min=%5.2f  med=%5.2f  p90=%5.2f  ns/op\n",
           label, s.min, s.med, s.p90);
}

// Bench the hash+bloom path: full qb_*_bulk(bytes, len, n) calls.
// Each repeat builds a fresh filter so insert isn't measuring an
// already-populated one. Mirrors harness.py benchmark().
static void bench_hash_bloom(size_t nbits, size_t n_insert, size_t n_query,
                             const uint8_t* keys_in, const uint8_t* keys_un,
                             stats_t* out_ins, stats_t* out_miss, stats_t* out_hit) {
    uint64_t ins[REPEATS], miss[REPEATS], hit[REPEATS];
    size_t sink = 0;
    for (int r = 0; r < WARMUP + REPEATS; r++) {
        void* f = qb_new(nbits);
        if (!f) { perror("qb_new"); exit(1); }
        uint64_t t0 = now_ns();
        qb_insert_bulk(f, keys_in, KLEN, n_insert);
        uint64_t t1 = now_ns();
        sink += qb_contains_bulk(f, keys_un, KLEN, n_query);
        uint64_t t2 = now_ns();
        sink += qb_contains_bulk(f, keys_in, KLEN, n_insert);
        uint64_t t3 = now_ns();
        qb_free(f);
        if (r >= WARMUP) {
            int i = r - WARMUP;
            ins[i]  = t1 - t0;
            miss[i] = t2 - t1;
            hit[i]  = t3 - t2;
        }
    }
    if (sink == 0xDEADBEEFDEADBEEFULL) printf("(sink hit; impossible)\n");
    *out_ins  = summarize(ins,  REPEATS, n_insert);
    *out_miss = summarize(miss, REPEATS, n_query);
    *out_hit  = summarize(hit,  REPEATS, n_insert);
}

// Bench the prehash path: qb_*_prehash_bulk on pre-computed uint64
// hashes. Separate timing loop so cache state matches harness.py
// benchmark_prehash() exactly.
static void bench_prehash(size_t nbits, size_t n_insert, size_t n_query,
                          const uint64_t* hash_in, const uint64_t* hash_un,
                          stats_t* out_ins, stats_t* out_miss, stats_t* out_hit) {
    uint64_t ins[REPEATS], miss[REPEATS], hit[REPEATS];
    size_t sink = 0;
    for (int r = 0; r < WARMUP + REPEATS; r++) {
        void* f = qb_new(nbits);
        if (!f) { perror("qb_new"); exit(1); }
        uint64_t t0 = now_ns();
        qb_insert_prehash_bulk(f, hash_in, n_insert);
        uint64_t t1 = now_ns();
        sink += qb_contains_prehash_bulk(f, hash_un, n_query);
        uint64_t t2 = now_ns();
        sink += qb_contains_prehash_bulk(f, hash_in, n_insert);
        uint64_t t3 = now_ns();
        qb_free(f);
        if (r >= WARMUP) {
            int i = r - WARMUP;
            ins[i]  = t1 - t0;
            miss[i] = t2 - t1;
            hit[i]  = t3 - t2;
        }
    }
    if (sink == 0xDEADBEEFDEADBEEFULL) printf("(sink hit; impossible)\n");
    *out_ins  = summarize(ins,  REPEATS, n_insert);
    *out_miss = summarize(miss, REPEATS, n_query);
    *out_hit  = summarize(hit,  REPEATS, n_insert);
}

static void run_size(const char* tag, size_t nbits,
                     size_t n_insert, size_t n_query) {
    // aligned_alloc requires size to be a multiple of alignment; round
    // each buffer up to the next 64-byte boundary via ROUND64.
    uint8_t* keys_in  = (uint8_t*)aligned_alloc(64, ROUND64(n_insert * KLEN));
    uint8_t* keys_un  = (uint8_t*)aligned_alloc(64, ROUND64(n_query  * KLEN));
    uint64_t* hash_in = (uint64_t*)aligned_alloc(64, ROUND64(n_insert * sizeof(uint64_t)));
    uint64_t* hash_un = (uint64_t*)aligned_alloc(64, ROUND64(n_query  * sizeof(uint64_t)));
    if (!keys_in || !keys_un || !hash_in || !hash_un) {
        perror("aligned_alloc"); exit(1);
    }
    fill_bytes(keys_in, n_insert * KLEN, 0xA1A1A1A1ULL);
    fill_bytes(keys_un, n_query  * KLEN, 0xB2B2B2B2ULL);
    fill_hashes(hash_in, n_insert, 0xC3C3C3C3ULL);
    fill_hashes(hash_un, n_query,  0xD4D4D4D4ULL);

    double mb = (double)(nbits / 8) / (1024.0 * 1024.0);
    double kb = (double)(nbits / 8) / 1024.0;
    if (mb >= 1.0) {
        printf("# %s: nbits=%zu (%.1f MB), n_insert=%zu, n_query=%zu\n",
               tag, nbits, mb, n_insert, n_query);
    } else {
        printf("# %s: nbits=%zu (%.0f KB), n_insert=%zu, n_query=%zu\n",
               tag, nbits, kb, n_insert, n_query);
    }

    stats_t ins, miss, hit, pins, pmiss, phit;
    bench_hash_bloom(nbits, n_insert, n_query, keys_in, keys_un,  &ins,  &miss,  &hit);
    bench_prehash   (nbits, n_insert, n_query, hash_in, hash_un, &pins, &pmiss, &phit);

    print_row("insert",         ins);
    print_row("miss",           miss);
    print_row("hit",            hit);
    print_row("prehash insert", pins);
    print_row("prehash miss",   pmiss);
    print_row("prehash hit",    phit);
    printf("\n");

    free(keys_in); free(keys_un); free(hash_in); free(hash_un);
}

// Filter sizes matching what harness.py picks on a 1 MB-L2 / 33 MB-L3
// host (the bench host the README is captured on). Override with the
// QB_NBITS / QB_N_INSERT env vars for one-off runs.
typedef struct { const char* tag; size_t nbits; size_t n_insert; size_t n_query; } preset_t;
static const preset_t PRESETS[] = {
    { "S",  1ULL << 20,    49932,   200000 },   //   1 Mbit (128 KB)
    { "M",  1ULL << 24,   798915,   500000 },   //  16 Mbit (  2 MB)
    { "L",  1ULL << 28, 12782640,   500000 },   // 256 Mbit ( 32 MB)
};

int main(int argc, char** argv) {
    printf("bench_qb: native C bench of qb_* kernels "
           "(repeats=%d, warmup=%d)\n\n", REPEATS, WARMUP);

    // Single-shot override.
    const char* env_nb = getenv("QB_NBITS");
    const char* env_ni = getenv("QB_N_INSERT");
    if (env_nb && env_ni) {
        size_t nb = strtoull(env_nb, NULL, 0);
        size_t ni = strtoull(env_ni, NULL, 0);
        run_size("custom", nb, ni, ni);
        return 0;
    }

    // CLI filter, e.g. `bench_qb S` or `bench_qb M L`.
    int want_all = (argc == 1);
    for (size_t i = 0; i < sizeof(PRESETS) / sizeof(PRESETS[0]); i++) {
        const preset_t* p = &PRESETS[i];
        int want = want_all;
        for (int a = 1; a < argc && !want; a++) {
            if (strcmp(argv[a], p->tag) == 0) want = 1;
        }
        if (want) run_size(p->tag, p->nbits, p->n_insert, p->n_query);
    }
    return 0;
}
