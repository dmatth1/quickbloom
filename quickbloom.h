// quickbloom.h -- public API for the quickbloom Bloom filter library.
//
// Three SBBF variants, all sharing the same single-key API shape. Pick
// the variant whose performance profile matches your workload:
//
//   qb_single_key_*  In-cache filters (< ~10 MB) or contains-heavy
//                    workloads. Lowest in-cache latency. No prefetch.
//
//   qb_unified_*     Good default. Adds software prefetch lookahead of
//                    8 keys; competitive across the cache hierarchy.
//                    Slight in-cache regression vs single_key.
//
//   qb_batched_*     Out-of-cache filters (> ~10 MB) or columnar/bulk
//                    workloads. 64-bit blocks, K=4, plus the 8-way
//                    batched ABI (qb_batched_*_batch8).
//
// All three variants link into the same libquickbloom.so / .a and can
// coexist in a single binary because their public symbols are
// namespaced.
//
// Algorithm: Split Block Bloom Filter, Apache Parquet spec (Putze,
// Sanders, Singler 2007). Same bit-position salt and 256-bit-block
// (or, for batched, 64-bit-block) layout used by arrow-cpp / arrow-rs
// / Velox / DuckDB.
//
// Thread safety:
//   - Concurrent reads (qb_*_contains, qb_*_contains_prehash,
//     qb_*_contains_bulk, qb_*_contains_prehash_bulk,
//     qb_batched_contains_batch8(_bulk)) are safe.
//   - Concurrent writes (qb_*_insert*) are NOT safe; the bit-set is
//     done with a non-atomic load-or-store. Callers that insert
//     concurrently must provide their own synchronization.
//   - Reads concurrent with writes may see partial state but never a
//     false negative once the write completes.
//
// CPU target: x86_64 with AVX2 + BMI2 (Intel Haswell 2013+, AMD
// Excavator 2015+ / Zen 1+). Compile with at least
// `-mavx2 -mbmi2 -mfma -maes`.

#ifndef QUICKBLOOM_H
#define QUICKBLOOM_H

#include <stddef.h>
#include <stdint.h>

#define QUICKBLOOM_VERSION_MAJOR 0
#define QUICKBLOOM_VERSION_MINOR 1
#define QUICKBLOOM_VERSION_PATCH 0
#define QUICKBLOOM_VERSION_STRING "0.1.0"

#ifdef __cplusplus
extern "C" {
#endif

// Single-key + prehash API shared by all three variants. NS is one of
// qb_single_key, qb_unified, qb_batched.
//
// Conventions:
//   - <NS>_new returns NULL on allocation failure.
//   - <NS>_free accepts NULL.
//   - <NS>_contains returns nonzero if the key may be present.
//   - *_bulk variants return the number of hits in the queried batch.
#define QB_DECLARE_CORE(NS) \
    void*  NS##_new(size_t nbits); \
    void   NS##_free(void* p); \
    void   NS##_insert(void* p, const void* key, size_t len); \
    int    NS##_contains(void* p, const void* key, size_t len); \
    void   NS##_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n); \
    size_t NS##_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n); \
    void   NS##_insert_prehash(void* p, uint64_t hash); \
    int    NS##_contains_prehash(void* p, uint64_t hash); \
    void   NS##_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n); \
    size_t NS##_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n);

QB_DECLARE_CORE(qb_single_key)
QB_DECLARE_CORE(qb_unified)
QB_DECLARE_CORE(qb_batched)

#undef QB_DECLARE_CORE

// 8-way batched ABI, batched variant only. Pass exactly 8 pre-computed
// 64-bit hashes; qb_batched_contains_batch8 returns a bitmap of the 8
// results in the low byte (bit i == 1 means hashes[i] may be present).
void    qb_batched_insert_batch8(void* p, const uint64_t hashes[8]);
uint8_t qb_batched_contains_batch8(void* p, const uint64_t hashes[8]);
void    qb_batched_insert_batch8_bulk(void* p, const uint64_t* hashes, size_t n);
size_t  qb_batched_contains_batch8_bulk(void* p, const uint64_t* hashes, size_t n);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // QUICKBLOOM_H
