// bloom_xorfuse.c -- binary fuse filter (Graf & Lemire 2020/22), shim
// to expose the bloom_* ABI used by the rest of the bench harness.
//
// Binary fuse is *not* a Bloom filter; it's a static-membership filter
// with much better space efficiency (~9 bits/key for ~0.4% FP) and
// generally faster contains() on read-only workloads. Source:
//   https://github.com/FastFilter/xor_singleheader  (Apache 2.0)
//
// The ABI mismatch we paper over:
//   - Bloom is dynamic: insert keys incrementally, contains is O(1) each.
//   - Binary fuse is static: populate(all_keys) once, then contains.
// This shim accumulates keys passed to bloom_insert* into an internal
// buffer and lazily calls binary_fuse8_populate on the first contains.
// Once populated, the filter is fixed and any subsequent inserts
// trigger a re-populate (slow). The bench harness inserts everything
// up-front in one bulk call, then probes, so this is fine for the
// bench but is NOT a drop-in for incremental workloads -- use a
// dynamic Bloom for that case.
//
// "insert" timing notes: bloom_insert_bulk / bloom_insert_prehash_bulk
// do the work upfront (hash + allocate + populate); bloom_insert /
// bloom_insert_prehash just append to the buffer and defer the build.
// Compare quickbloom's bloom_insert_bulk numbers against xorfuse's
// bloom_insert_bulk numbers for an apples-to-apples build-time view;
// compare bloom_contains* for query latency.

#define K_HASHES 3   // binary_fuse8 uses 3 fingerprint lookups per contain

#include "binaryfusefilter.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Same 16-byte fast hash quickbloom uses, so the bytes-in path is
// apples-to-apples on hashing cost.
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

// Wrapper holds the binary fuse filter plus a key buffer for the
// staged-insert/lazy-build flow.
typedef struct {
    binary_fuse8_t filter;
    uint64_t* buf;       // accumulated keys (as 64-bit hashes)
    size_t    buf_len;   // number of keys currently buffered
    size_t    buf_cap;   // capacity of the buffer
    int       built;     // 1 once binary_fuse8_populate has succeeded
} xf_t;

static void buf_push(xf_t* x, uint64_t h) {
    if (x->buf_len == x->buf_cap) {
        size_t new_cap = x->buf_cap ? x->buf_cap * 2 : 1024;
        x->buf = (uint64_t*)realloc(x->buf, new_cap * sizeof(uint64_t));
        x->buf_cap = new_cap;
    }
    x->buf[x->buf_len++] = h;
}

static void build(xf_t* x) {
    if (x->built) return;
    if (x->buf_len == 0) {
        // binary_fuse8 requires size >= 2 for a useful filter
        buf_push(x, 0); buf_push(x, 1);
    }
    binary_fuse8_allocate((uint32_t)x->buf_len, &x->filter);
    binary_fuse8_populate(x->buf, (uint32_t)x->buf_len, &x->filter);
    x->built = 1;
}

void* bloom_new(size_t nbits) {
    xf_t* x = (xf_t*)calloc(1, sizeof(xf_t));
    if (!x) return NULL;
    // nbits gives us an upper bound on the expected key count: at the
    // 1% bloom sizing this is ~21 bits/key so reserve ~nbits/21 keys.
    size_t reserve = nbits / 21;
    if (reserve < 1024) reserve = 1024;
    x->buf = (uint64_t*)malloc(reserve * sizeof(uint64_t));
    x->buf_cap = x->buf ? reserve : 0;
    return x;
}

void bloom_free(void* p) {
    if (!p) return;
    xf_t* x = (xf_t*)p;
    if (x->built) binary_fuse8_free(&x->filter);
    free(x->buf);
    free(x);
}

void bloom_insert_prehash(void* p, uint64_t h) {
    xf_t* x = (xf_t*)p;
    // If we already built, we'd have to rebuild on every insert --
    // mark not-built so the next contains will rebuild from the
    // updated buffer. This is intentionally slow; xor filters are
    // not designed for incremental insert.
    if (x->built) { binary_fuse8_free(&x->filter); x->built = 0; }
    buf_push(x, h);
}

int bloom_contains_prehash(void* p, uint64_t h) {
    xf_t* x = (xf_t*)p;
    if (!x->built) build(x);
    return binary_fuse8_contain(h, &x->filter);
}

void bloom_insert(void* p, const void* key, size_t len) {
    bloom_insert_prehash(p, bloom_hash(key, len));
}

int bloom_contains(void* p, const void* key, size_t len) {
    return bloom_contains_prehash(p, bloom_hash(key, len));
}

// Bulk paths: accumulate all hashes, then build once. This is the
// happy path for xor filters.
void bloom_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    xf_t* x = (xf_t*)p;
    if (x->built) { binary_fuse8_free(&x->filter); x->built = 0; }
    // Reserve space.
    if (x->buf_cap < x->buf_len + n) {
        size_t new_cap = x->buf_cap;
        while (new_cap < x->buf_len + n) new_cap = new_cap ? new_cap * 2 : 1024;
        x->buf = (uint64_t*)realloc(x->buf, new_cap * sizeof(uint64_t));
        x->buf_cap = new_cap;
    }
    for (size_t i = 0; i < n; i++) {
        x->buf[x->buf_len + i] = bloom_hash(keys + i * klen, klen);
    }
    x->buf_len += n;
    build(x);
}

size_t bloom_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n) {
    xf_t* x = (xf_t*)p;
    if (!x->built) build(x);
    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        if (binary_fuse8_contain(bloom_hash(keys + i * klen, klen), &x->filter)) hits++;
    }
    return hits;
}

void bloom_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    xf_t* x = (xf_t*)p;
    if (x->built) { binary_fuse8_free(&x->filter); x->built = 0; }
    if (x->buf_cap < x->buf_len + n) {
        size_t new_cap = x->buf_cap;
        while (new_cap < x->buf_len + n) new_cap = new_cap ? new_cap * 2 : 1024;
        x->buf = (uint64_t*)realloc(x->buf, new_cap * sizeof(uint64_t));
        x->buf_cap = new_cap;
    }
    memcpy(x->buf + x->buf_len, hashes, n * sizeof(uint64_t));
    x->buf_len += n;
    build(x);
}

size_t bloom_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n) {
    xf_t* x = (xf_t*)p;
    if (!x->built) build(x);
    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        if (binary_fuse8_contain(hashes[i], &x->filter)) hits++;
    }
    return hits;
}
