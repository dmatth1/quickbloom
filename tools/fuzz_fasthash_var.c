// fuzz_fasthash_var.c -- libFuzzer harness focused on the
// fasthash64_var tail switch (1..7 byte tails after the 8-byte blocks).
//
// fasthash64_var lives inside quickbloom.c as static inline, but we
// reach it via the public qb_insert path for any key whose length is
// not exactly 16 bytes. This harness biases toward 1..23 byte keys
// so the fuzzer concentrates on lengths the 16-byte fast path
// doesn't handle.
//
// Build with `make fuzz CC=clang`. Run:
//   ./build/fuzz_fasthash_var -max_total_time=60

#include "quickbloom.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) return 0;
    // klen biased to 1..23 to exercise every tail-switch case
    // (lengths mod 8 == 1..7) plus the small-input branches.
    size_t klen = ((size_t)data[0] % 23) + 1;
    data += 1; size -= 1;

    size_t max_keys = size / klen;
    if (max_keys == 0) return 0;
    if (max_keys > 2048) max_keys = 2048;

    void* f = qb_new(qb_estimate_bits(max_keys, 0.01));
    if (!f) return 0;

    for (size_t i = 0; i < max_keys; i++) {
        qb_insert(f, data + i * klen, klen);
    }
    for (size_t i = 0; i < max_keys; i++) {
        if (!qb_contains(f, data + i * klen, klen)) __builtin_trap();
    }

    qb_free(f);
    return 0;
}
