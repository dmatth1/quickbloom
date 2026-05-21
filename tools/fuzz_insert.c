// fuzz_insert.c -- libFuzzer harness for qb_insert_bulk.
//
// Feeds the fuzzer-provided input as a packed buffer of fixed-length
// keys, calls insert_bulk + contains_bulk + serialize round-trip,
// asserts the no-false-negatives invariant. Build with:
//   make fuzz CC=clang
//
// Run with:
//   ./build/fuzz_insert -max_total_time=60

#include "quickbloom.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16) return 0;
    // Pick a klen from the fuzzer's first byte, bounded to 1..64.
    size_t klen = (data[0] % 64) + 1;
    data += 1; size -= 1;
    if (size < klen) return 0;
    size_t n = size / klen;
    if (n == 0) return 0;

    // Cap n so the fuzzer doesn't OOM on us.
    if (n > 4096) n = 4096;

    // Filter sized comfortably for n keys at ~1% FP.
    void* f = qb_new(qb_estimate_bits(n, 0.01));
    if (!f) return 0;

    qb_insert_bulk(f, data, klen, n);

    // No false negatives: every inserted key must be reported present.
    for (size_t i = 0; i < n; i++) {
        if (!qb_contains(f, data + i * klen, klen)) {
            __builtin_trap();  // false negative -- algorithm bug
        }
    }

    // Bulk contains must report >= n hits on the inserted set.
    size_t hits = qb_contains_bulk(f, data, klen, n);
    if (hits < n) __builtin_trap();

    // Serialize round-trip must preserve membership.
    size_t nbytes = qb_serialized_size(f);
    uint8_t* buf = (uint8_t*)malloc(nbytes);
    if (buf) {
        qb_serialize(f, buf);
        void* g = qb_deserialize(buf, nbytes);
        if (g) {
            for (size_t i = 0; i < n; i++) {
                if (!qb_contains(g, data + i * klen, klen)) __builtin_trap();
            }
            qb_free(g);
        }
        free(buf);
    }

    qb_free(f);
    return 0;
}
