// Parquet-spec-compliant SBBF with three probe paths side-by-side.
//
// All three operate on the SAME underlying byte array (Parquet on-disk
// layout: 256-bit blocks, 8 SALT-derived bit positions per block).
// Insertions go through one shared `insert` so the bit content is
// identical regardless of which probe path measures it. Probe results
// must agree bit-for-bit on any input -- enforced in the bench's
// diff-test phase.
//
// Hash: Parquet mandates XXH64-seed-0. We compute it once per inserted
// key / query value and hold the resulting uint64_t constant across the
// three paths so the comparison isolates the probe layout.
#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace parquet_bloom {

// Apache Parquet SBBF SALT constants. Verbatim from the spec and from
// the three reference implementations (arrow-cc, arrow-rs, velox).
alignas(32) inline constexpr uint32_t SALT[8] = {
    0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
    0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U,
};

inline constexpr int kBitsSetPerBlock = 8;
inline constexpr int kBytesPerFilterBlock = 32;  // 256 bits

class filter {
public:
    // num_bytes must be a multiple of 32 (one 256-bit block).
    explicit filter(size_t num_bytes);

    void insert(uint64_t hash);

    // ---- The three probe paths. All bit-identical on output. ----

    // (1) Scalar: line-for-line port of Apache Arrow C++'s
    //     BlockSplitBloomFilter::FindHash. The baseline shipped by
    //     arrow-cpp, arrow-rs, and Velox.
    bool find_scalar(uint64_t hash) const;

    // (2) xsimd drop-in on the same Parquet-spec layout. Bit-identical
    //     results. Uses xsimd::batch<uint32_t> (8 lanes on AVX2 / AVX-512;
    //     two 4-lane batches on NEON; selected per build target).
    bool find_simd(uint64_t hash) const;

    // (3) xsimd + 4-way unroll across row-group filters: probe 4 filters
    //     in parallel against the SAME query hash. Lets the OoO core have
    //     4 cache misses in flight simultaneously when filters are out of
    //     L3. Returns count of positives (0..4).
    static int find_simd_x4(const filter* f0, const filter* f1,
                            const filter* f2, const filter* f3,
                            uint64_t hash);

    size_t num_bytes() const { return data_.size(); }
    size_t num_blocks() const { return data_.size() / kBytesPerFilterBlock; }

private:
    // Stored as raw bytes so the scalar and SIMD paths see the same
    // memory layout the on-disk Parquet bloom does.
    std::vector<uint8_t> data_;
};

}  // namespace parquet_bloom
