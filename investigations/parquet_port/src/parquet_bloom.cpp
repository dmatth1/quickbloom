#include "parquet_bloom.h"
#include <cassert>
#include <cstring>
#include <immintrin.h>

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

// (2) AVX2 drop-in. Same Parquet-spec layout, same SALT order, same
//     bit positions. The mask compute `vpmulld + vpsrld + vpsllvd`
//     produces 8 lanes each equal to `1 << ((key * SALT[i]) >> 27)`
//     -- identical to the scalar inner loop. `testc` returns 1 iff
//     every bit in `mask` is also set in the loaded block.
bool filter::find_avx2(uint64_t hash) const {
    uint32_t bucket_index = uint32_t(
        ((hash >> 32) * (data_.size() / kBytesPerFilterBlock)) >> 32);
    uint32_t key = uint32_t(hash);

    __m256i hash_v = _mm256_set1_epi32(int32_t(key));
    __m256i salt   = _mm256_load_si256(reinterpret_cast<const __m256i*>(SALT));
    __m256i prod   = _mm256_mullo_epi32(hash_v, salt);
    __m256i shift  = _mm256_srli_epi32(prod, 27);
    __m256i ones   = _mm256_set1_epi32(1);
    __m256i mask   = _mm256_sllv_epi32(ones, shift);

    const __m256i* blk = reinterpret_cast<const __m256i*>(
        data_.data() + bucket_index * kBytesPerFilterBlock);
    __m256i blk_v = _mm256_loadu_si256(blk);
    return _mm256_testc_si256(blk_v, mask);
}

// (3) v11-style 4-way unrolled probe. Each filter is independent so all
//     four cache lines can be in flight at once; the OoO core overlaps
//     the cache-miss latency that dominates the out-of-L3 regime.
//     Same mask used for all four (same query hash).
int filter::find_avx2_x4(const filter* f0, const filter* f1,
                         const filter* f2, const filter* f3, uint64_t hash) {
    uint32_t key = uint32_t(hash);
    __m256i hash_v = _mm256_set1_epi32(int32_t(key));
    __m256i salt   = _mm256_load_si256(reinterpret_cast<const __m256i*>(SALT));
    __m256i prod   = _mm256_mullo_epi32(hash_v, salt);
    __m256i shift  = _mm256_srli_epi32(prod, 27);
    __m256i ones   = _mm256_set1_epi32(1);
    __m256i mask   = _mm256_sllv_epi32(ones, shift);

    auto bucket = [&](const filter* f) {
        return uint32_t(((hash >> 32) * (f->data_.size() / kBytesPerFilterBlock)) >> 32);
    };
    const __m256i* p0 = reinterpret_cast<const __m256i*>(
        f0->data_.data() + bucket(f0) * kBytesPerFilterBlock);
    const __m256i* p1 = reinterpret_cast<const __m256i*>(
        f1->data_.data() + bucket(f1) * kBytesPerFilterBlock);
    const __m256i* p2 = reinterpret_cast<const __m256i*>(
        f2->data_.data() + bucket(f2) * kBytesPerFilterBlock);
    const __m256i* p3 = reinterpret_cast<const __m256i*>(
        f3->data_.data() + bucket(f3) * kBytesPerFilterBlock);

    __m256i v0 = _mm256_loadu_si256(p0);
    __m256i v1 = _mm256_loadu_si256(p1);
    __m256i v2 = _mm256_loadu_si256(p2);
    __m256i v3 = _mm256_loadu_si256(p3);

    return _mm256_testc_si256(v0, mask) + _mm256_testc_si256(v1, mask)
         + _mm256_testc_si256(v2, mask) + _mm256_testc_si256(v3, mask);
}

}  // namespace parquet_bloom
