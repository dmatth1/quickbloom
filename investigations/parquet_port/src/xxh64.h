// Minimal XXH64 implementation matching the Parquet spec
// (https://github.com/apache/parquet-format/blob/master/BloomFilter.md
//  mandates xxHash with seed 0, spec version 0.1.1).
//
// Verified against the canonical Cyan4973/xxHash test vectors at startup.
// Not optimized -- the bench measures the PROBE path, not the hash. Holding
// the hash constant across implementations is what isolates the comparison.
#pragma once
#include <cstddef>
#include <cstdint>

namespace xxh {

uint64_t xxh64(const void* data, size_t len, uint64_t seed = 0);

// Sanity-checks against published XXH64 vectors. Returns true on success.
// Called once at bench startup so any mismatch aborts loudly before any
// timing data is collected.
bool self_test();

}  // namespace xxh
