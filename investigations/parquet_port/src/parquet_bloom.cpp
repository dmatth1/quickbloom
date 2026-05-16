#include "parquet_bloom.h"
#include <cassert>
#include <cstring>
#include <xsimd/xsimd.hpp>

namespace parquet_bloom {

filter::filter(size_t num_bytes) : data_(num_bytes, 0) {
    assert(num_bytes % kBytesPerFilterBlock == 0);
    assert(num_bytes > 0);
}

// Insert is the same algorithm as scalar find -- write the 8 bits in the
// block addressed by the high 32 bits of the hash. Both probe paths read
// the same bits afterward.
void filter::insert(uint64_t hash) {
    uint32_t bucket_index = uint32_t(
        ((hash >> 32) * (data_.size() / kBytesPerFilterBlock)) >> 32);
    uint32_t key = uint32_t(hash);
    uint32_t* bitset32 = reinterpret_cast<uint32_t*>(data_.data());
    for (int i = 0; i < kBitsSetPerBlock; i++) {
        uint32_t mask = uint32_t(1) << ((key * SALT[i]) >> 27);
        bitset32[kBitsSetPerBlock * bucket_index + i] |= mask;
    }
}

// (1) Scalar: line-for-line port of Apache Arrow C++'s FindHash.
//     Source: apache/arrow/cpp/src/parquet/bloom_filter.cc:348.
bool filter::find_scalar(uint64_t hash) const {
    uint32_t bucket_index = uint32_t(
        ((hash >> 32) * (data_.size() / kBytesPerFilterBlock)) >> 32);
    uint32_t key = uint32_t(hash);
    const uint32_t* bitset32 = reinterpret_cast<const uint32_t*>(data_.data());
    for (int i = 0; i < kBitsSetPerBlock; ++i) {
        uint32_t mask = uint32_t(1) << ((key * SALT[i]) >> 27);
        if (0 == (bitset32[kBitsSetPerBlock * bucket_index + i] & mask)) {
            return false;
        }
    }
    return true;
}

// (2) xsimd drop-in. Same Parquet-spec layout, same SALT order, same bit
//     positions. The mask compute uses xsimd::batch<uint32_t> ops which
//     lower to vpmulld + vpsrld + vpsllvd on AVX2 (and to the equivalent
//     NEON/AVX-512 sequences on those targets).
//
//     The "all mask bits set in block" reduction is one place xsimd's
//     public API costs us: `xsimd::all((blk & mask) == mask)` lowers
//     to vpand + vpcmpeqd + vpcmpeqd(allones) + vptest + setb (5 ops)
//     whereas the natural primitive is a single `_mm256_testc_si256`.
//     xsimd uses testc internally for `all(batch_bool)` but doesn't
//     expose `testc(batch, batch)` publicly. We reach for the raw
//     intrinsic here and let other archs fall back to the portable
//     reduction. (A proper fix is a small xsimd PR adding
//     `xsimd::testc(batch, batch)` as a portable primitive.)
namespace {

using batch_t = xsimd::batch<uint32_t>;
static_assert(batch_t::size == kBitsSetPerBlock,
              "Parquet SBBF expects an 8-lane SIMD batch; this build's "
              "default xsimd batch isn't 8 wide. AVX2/AVX-512 builds get "
              "this for free; a NEON port needs a different splitting.");

inline batch_t simd_mask(uint32_t key) {
    batch_t hash_v = batch_t::broadcast(key);
    batch_t salt   = batch_t::load_aligned(SALT);
    batch_t prod   = hash_v * salt;
    batch_t shift  = prod >> 27;
    batch_t ones   = batch_t::broadcast(uint32_t(1));
    return ones << shift;
}

inline batch_t load_block(const uint8_t* data, uint32_t bucket_index) {
    return batch_t::load_unaligned(reinterpret_cast<const uint32_t*>(
        data + bucket_index * kBytesPerFilterBlock));
}

// "Are all bits set in `mask` also set in `blk`?" -- one vptest on AVX2.
inline bool all_mask_bits_set(batch_t blk, batch_t mask) {
#if defined(__AVX__)
    return _mm256_testc_si256(__m256i(blk), __m256i(mask)) != 0;
#else
    // Portable fallback. xsimd::all(...) is currently 5 ops on AVX2,
    // but on other archs (NEON, AVX-512) it's already the natural shape.
    return xsimd::all((blk & mask) == mask);
#endif
}

}  // namespace

bool filter::find_simd(uint64_t hash) const {
    uint32_t bucket_index = uint32_t(
        ((hash >> 32) * (data_.size() / kBytesPerFilterBlock)) >> 32);
    uint32_t key = uint32_t(hash);

    batch_t mask = simd_mask(key);
    batch_t blk  = load_block(data_.data(), bucket_index);
    return all_mask_bits_set(blk, mask);
}

// (3) 4-way unrolled probe across row-group filters. Each filter is
//     independent so all four cache lines can be in flight at once; the
//     OoO core overlaps the cache-miss latency that dominates the
//     out-of-L3 regime. Same mask used for all four (same query hash).
int filter::find_simd_x4(const filter* f0, const filter* f1,
                         const filter* f2, const filter* f3, uint64_t hash) {
    uint32_t key = uint32_t(hash);
    batch_t mask = simd_mask(key);

    auto bucket = [&](const filter* f) {
        return uint32_t(((hash >> 32) * (f->data_.size() / kBytesPerFilterBlock)) >> 32);
    };

    batch_t v0 = load_block(f0->data_.data(), bucket(f0));
    batch_t v1 = load_block(f1->data_.data(), bucket(f1));
    batch_t v2 = load_block(f2->data_.data(), bucket(f2));
    batch_t v3 = load_block(f3->data_.data(), bucket(f3));

    return int(all_mask_bits_set(v0, mask))
         + int(all_mask_bits_set(v1, mask))
         + int(all_mask_bits_set(v2, mask))
         + int(all_mask_bits_set(v3, mask));
}

}  // namespace parquet_bloom
