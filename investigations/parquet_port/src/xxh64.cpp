#include "xxh64.h"
#include <bit>
#include <cstring>

namespace xxh {

static constexpr uint64_t P1 = 0x9E3779B185EBCA87ULL;
static constexpr uint64_t P2 = 0xC2B2AE3D27D4EB4FULL;
static constexpr uint64_t P3 = 0x165667B19E3779F9ULL;
static constexpr uint64_t P4 = 0x85EBCA77C2B2AE63ULL;
static constexpr uint64_t P5 = 0x27D4EB2F165667C5ULL;

static inline uint64_t read_u64(const void* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}
static inline uint32_t read_u32(const void* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * P2;
    acc = std::rotl(acc, 31);
    acc *= P1;
    return acc;
}

static inline uint64_t merge_round(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * P1 + P4;
    return acc;
}

uint64_t xxh64(const void* data_, size_t len, uint64_t seed) {
    const uint8_t* p = static_cast<const uint8_t*>(data_);
    const uint8_t* end = p + len;
    uint64_t h;

    if (len >= 32) {
        const uint8_t* limit = end - 32;
        uint64_t v1 = seed + P1 + P2;
        uint64_t v2 = seed + P2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - P1;
        do {
            v1 = round64(v1, read_u64(p));     p += 8;
            v2 = round64(v2, read_u64(p));     p += 8;
            v3 = round64(v3, read_u64(p));     p += 8;
            v4 = round64(v4, read_u64(p));     p += 8;
        } while (p <= limit);
        h = std::rotl(v1, 1) + std::rotl(v2, 7) + std::rotl(v3, 12) + std::rotl(v4, 18);
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = seed + P5;
    }

    h += uint64_t(len);

    while (p + 8 <= end) {
        uint64_t k = round64(0, read_u64(p));
        h ^= k;
        h = std::rotl(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= uint64_t(read_u32(p)) * P1;
        h = std::rotl(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= uint64_t(*p) * P5;
        h = std::rotl(h, 11) * P1;
        p++;
    }

    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

bool self_test() {
    // Canonical XXH64 test vectors from Cyan4973/xxHash documentation.
    // Seed = 0 (Parquet spec default).
    struct V { const char* s; size_t n; uint64_t expect; };
    const V cases[] = {
        { "",         0, 0xEF46DB3751D8E999ULL },
        { "a",        1, 0xD24EC4F1A98C6E5BULL },
        { "abc",      3, 0x44BC2CF5AD770999ULL },
    };
    for (const auto& c : cases) {
        if (xxh64(c.s, c.n, 0) != c.expect) return false;
    }
    return true;
}

}  // namespace xxh
