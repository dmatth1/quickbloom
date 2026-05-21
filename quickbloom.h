// quickbloom.h -- public API for the quickbloom Bloom filter library.
//
// One implementation: a fast Split Block Bloom Filter (SBBF) with
// wymum hashing and AVX2 SIMD mask compute. The fastest single-key
// SBBF probe kernel on AVX2 x86_64 across all three cache regimes we
// benchmark — see README's Performance section for the comparison.
//
// Algorithm: SBBF, Apache Parquet spec (Putze, Sanders, Singler 2007).
// 256-bit blocks, K=8 bits per probe, all bits in a single cache line.
// Bit-identical bitset to other Parquet SBBF implementations
// (arrow-cpp, arrow-rs, Velox, DuckDB, Impala) for the same 64-bit
// hash input.
//
// Thread safety:
//   - Concurrent reads (qb_contains, qb_contains_prehash,
//     qb_contains_bulk, qb_contains_prehash_bulk) are safe.
//   - Concurrent writes (qb_insert*) are NOT safe; the block-level
//     bit-set is done with a non-atomic load-or-store. Callers that
//     insert concurrently must provide their own synchronization.
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

// Lifecycle.
//   qb_new returns NULL on allocation failure.
//   qb_free accepts NULL.
void*  qb_new(size_t nbits);
void   qb_free(void* p);

// Single-key API.
//   qb_contains returns nonzero if the key may be present.
void   qb_insert(void* p, const void* key, size_t len);
int    qb_contains(void* p, const void* key, size_t len);

// Bulk API. Returns the number of hits in the queried batch.
void   qb_insert_bulk(void* p, const uint8_t* keys, size_t klen, size_t n);
size_t qb_contains_bulk(void* p, const uint8_t* keys, size_t klen, size_t n);

// Prehash API. Callers that already have a 64-bit hash for the key
// can skip the built-in wymum step.
void   qb_insert_prehash(void* p, uint64_t hash);
int    qb_contains_prehash(void* p, uint64_t hash);
void   qb_insert_prehash_bulk(void* p, const uint64_t* hashes, size_t n);
size_t qb_contains_prehash_bulk(void* p, const uint64_t* hashes, size_t n);

// Sizing helper. Returns a recommended filter size in bits to hold
// approximately n items at the requested false-positive rate fp. Pass
// the result to qb_new; the filter rounds it up to a power of two
// internally.
//
// The formula is the classical Bloom filter bit-budget; SBBF's actual
// FP rate at this sizing is typically within ~2x of fp on small
// filters and converges to fp as the filter grows. Callers who need
// a specific FP target on a small filter should over-size (e.g.
// multiply by 1.5).
size_t qb_estimate_bits(size_t n, double fp);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // QUICKBLOOM_H
