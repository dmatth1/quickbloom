// Bench: Apache Arrow C++ / arrow-rs / Velox scalar Parquet bloom probe
// vs AVX2 probe on the SAME Parquet-spec layout.
//
// Workload: simulates a SELECT * WHERE col = 'X' or IN (...) scan over an
// N-row-group Parquet table. For each query value:
//   1. Compute XXH64 (held constant across both probes)
//   2. Probe N row-group blooms in sequence
// Both probe paths see the same on-disk bit content; only the inner loop
// differs. A diff-test phase guarantees bit-identical results.
//
// Two regimes:
//   small  : 64 row groups * 8 KiB each   = 512 KiB total  (in L2)
//   medium : 4096 row groups * 32 KiB each = 128 MiB total (out of L3)
//   large  : 16384 row groups * 64 KiB each = 1 GiB total  (deep DRAM)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "parquet_bloom.h"
#include "xxh64.h"

using clk = std::chrono::steady_clock;
static inline double ns_since(clk::time_point t) {
    return std::chrono::duration<double, std::nano>(clk::now() - t).count();
}

// Build N filters of the given byte size, populate them with `keys_per_filter`
// random 16-byte keys (different per filter), return alongside a vector of
// pre-hashed query keys (mix of hit + miss).
struct Setup {
    std::vector<parquet_bloom::filter> filters;
    std::vector<uint64_t> query_hashes;  // miss-mostly workload
};

static Setup build(int n_filters, size_t filter_bytes,
                   size_t keys_per_filter, size_t n_queries, uint64_t seed) {
    Setup s;
    s.filters.reserve(n_filters);
    std::mt19937_64 rng(seed);

    uint8_t key[16];
    for (int f = 0; f < n_filters; f++) {
        s.filters.emplace_back(filter_bytes);
        for (size_t k = 0; k < keys_per_filter; k++) {
            uint64_t a = rng(), b = rng();
            std::memcpy(key + 0, &a, 8);
            std::memcpy(key + 8, &b, 8);
            uint64_t h = xxh::xxh64(key, 16, 0);
            s.filters.back().insert(h);
        }
    }
    s.query_hashes.reserve(n_queries);
    for (size_t q = 0; q < n_queries; q++) {
        // Bias to a half-space disjoint from inserted keys -> mostly miss.
        uint64_t a = rng() | 0x8000000000000000ULL, b = rng();
        std::memcpy(key + 0, &a, 8);
        std::memcpy(key + 8, &b, 8);
        s.query_hashes.push_back(xxh::xxh64(key, 16, 0));
    }
    return s;
}

// Diff-test: every (query, filter) pair must produce the same answer
// from the scalar and AVX2 paths. Catches any layout/mask bug.
static void diff_test(const Setup& s) {
    size_t total = 0, mismatches = 0;
    for (uint64_t h : s.query_hashes) {
        for (auto& f : s.filters) {
            bool a = f.find_scalar(h);
            bool b = f.find_avx2(h);
            if (a != b) mismatches++;
            total++;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr, "DIFF FAIL: %zu / %zu mismatches\n", mismatches, total);
        std::exit(1);
    }
    fprintf(stdout, "  diff-test: %zu probes, 0 mismatches.\n", total);
}

struct Result { double scalar_ns, avx2_ns, avx2x4_ns; };

static Result bench_one(const Setup& s, int repeats = 5) {
    std::vector<double> scalar_ns, avx2_ns, avx2x4_ns;
    int nf = int(s.filters.size());
    for (int rep = 0; rep < repeats + 1; rep++) {
        auto t = clk::now();
        volatile size_t hits1 = 0;
        for (uint64_t h : s.query_hashes) {
            for (auto& f : s.filters) hits1 += f.find_scalar(h);
        }
        double s_ns = ns_since(t);

        t = clk::now();
        volatile size_t hits2 = 0;
        for (uint64_t h : s.query_hashes) {
            for (auto& f : s.filters) hits2 += f.find_avx2(h);
        }
        double a_ns = ns_since(t);

        // 4-way unroll: probe 4 filters in parallel against same hash.
        t = clk::now();
        volatile size_t hits3 = 0;
        for (uint64_t h : s.query_hashes) {
            int i = 0;
            for (; i + 4 <= nf; i += 4) {
                hits3 += parquet_bloom::filter::find_avx2_x4(
                    &s.filters[i], &s.filters[i+1],
                    &s.filters[i+2], &s.filters[i+3], h);
            }
            for (; i < nf; i++) hits3 += s.filters[i].find_avx2(h);
        }
        double x4_ns = ns_since(t);

        (void)hits1; (void)hits2; (void)hits3;
        if (rep > 0) {
            scalar_ns.push_back(s_ns);
            avx2_ns.push_back(a_ns);
            avx2x4_ns.push_back(x4_ns);
        }
    }
    std::sort(scalar_ns.begin(), scalar_ns.end());
    std::sort(avx2_ns.begin(),   avx2_ns.end());
    std::sort(avx2x4_ns.begin(), avx2x4_ns.end());
    return { scalar_ns[scalar_ns.size()/2],
             avx2_ns[avx2_ns.size()/2],
             avx2x4_ns[avx2x4_ns.size()/2] };
}

static void run_regime(const char* label, int n_filters, size_t filter_bytes,
                       size_t keys_per_filter, size_t n_queries, uint64_t seed) {
    printf("\n=== %s ===\n", label);
    size_t total_bytes = size_t(n_filters) * filter_bytes;
    printf("  %d row groups x %zu KiB each = %.1f MiB total filter footprint\n",
           n_filters, filter_bytes / 1024, total_bytes / (1024.0 * 1024.0));
    printf("  %zu keys per filter, %zu query values\n", keys_per_filter, n_queries);

    Setup s = build(n_filters, filter_bytes, keys_per_filter, n_queries, seed);
    diff_test(s);
    Result r = bench_one(s);

    size_t total_probes = s.query_hashes.size() * size_t(n_filters);
    double scalar_per = r.scalar_ns / double(total_probes);
    double avx2_per   = r.avx2_ns   / double(total_probes);
    double x4_per     = r.avx2x4_ns / double(total_probes);
    printf("  scalar (arrow-cpp/arrow-rs/velox shape):       %.2f ns/probe\n", scalar_per);
    printf("  AVX2 single-key (drop-in, same layout):        %.2f ns/probe\n", avx2_per);
    printf("  AVX2 4-way unroll across row groups:           %.2f ns/probe\n", x4_per);
    printf("  -> %.2fx single-key, %.2fx with 4-way across row groups\n",
           scalar_per / avx2_per, scalar_per / x4_per);

    printf("  per simulated `WHERE col = 'X'` query (= %d row group probes):\n",
           n_filters);
    printf("    scalar: %.2f us  /  AVX2: %.2f us  /  AVX2-x4: %.2f us  /  savings (best): %.2f us\n",
           scalar_per * n_filters / 1000.0,
           avx2_per   * n_filters / 1000.0,
           x4_per     * n_filters / 1000.0,
           (scalar_per - x4_per) * n_filters / 1000.0);
}

int main() {
    setbuf(stdout, nullptr);
    printf("Parquet bloom port: scalar (arrow-cpp / arrow-rs / velox shape)\n");
    printf("                  vs AVX2 drop-in on the same Parquet-spec layout\n");

    if (!xxh::self_test()) {
        fprintf(stderr, "XXH64 self-test FAILED -- aborting.\n");
        return 1;
    }
    printf("XXH64 self-test: OK\n");

    // Small: filter fits in L1/L2.
    run_regime("small (in-cache)",
               /*n_filters*/ 64, /*filter_bytes*/ 8 * 1024,
               /*keys/filter*/ 4096, /*queries*/ 50'000, 0xC0FFEE);

    // Medium: 128 MiB total, out of L3 on this box.
    run_regime("medium (out-of-L3, 128 MiB)",
               /*n_filters*/ 4096, /*filter_bytes*/ 32 * 1024,
               /*keys/filter*/ 16'000, /*queries*/ 20'000, 0xC0FFEE);

    // Large: 1 GiB filter footprint -- production-shape big table scan.
    run_regime("large (deep out-of-L3, 1 GiB)",
               /*n_filters*/ 16384, /*filter_bytes*/ 64 * 1024,
               /*keys/filter*/ 32'000, /*queries*/ 5'000, 0xC0FFEE);

    return 0;
}
