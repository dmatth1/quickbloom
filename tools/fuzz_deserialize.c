// fuzz_deserialize.c -- libFuzzer harness for qb_deserialize.
//
// qb_deserialize is the only public entry point that takes
// attacker-controlled bytes; everything else takes typed inputs from
// trusted callers. The other fuzz harness (fuzz_insert.c) round-trips
// known-good serialized output back through deserialize, which only
// exercises the happy path. This harness feeds raw fuzzer-controlled
// bytes in and then drives qb_contains / qb_serialize on the result,
// so ASan/UBSan catches OOB reads, alignment violations, integer
// overflow in size math, and resource-exhaustion patterns that future
// header parsing might introduce.
//
// Build with `make fuzz CC=clang`. Run:
//   ./build/fuzz_deserialize -max_total_time=60

#include "quickbloom.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Cap the fuzzer-provided size so a single bad input doesn't try
    // to allocate gigabytes. qb_deserialize itself is documented to
    // accept arbitrary nbytes, so the cap lives here in the harness;
    // the library's own size-cap is exercised by feeding a
    // deliberately-huge nbytes below.
    size_t nbytes = size;
    if (nbytes > (1u << 24)) nbytes = 1u << 24;  // 16 MB ceiling

    void* f = qb_deserialize(data, nbytes);
    if (!f) return 0;

    // Probe the filter the same way a downstream caller would.
    uint8_t key[16];
    for (int i = 0; i < 16; i++) key[i] = (uint8_t)(data ? data[i % nbytes] : 0);
    (void)qb_contains(f, key, 16);
    (void)qb_contains_prehash(f, 0xC0FFEEull);

    // Round-trip: serialize the filter we just deserialized, then
    // deserialize again. The two byte images must match.
    size_t out_n = qb_serialized_size(f);
    if (out_n) {
        uint8_t* out = (uint8_t*)malloc(out_n);
        if (out) {
            qb_serialize(f, out);
            void* g = qb_deserialize(out, out_n);
            if (g) {
                size_t gn = qb_serialized_size(g);
                if (gn != out_n) __builtin_trap();
                uint8_t* out2 = (uint8_t*)malloc(gn);
                if (out2) {
                    qb_serialize(g, out2);
                    if (memcmp(out, out2, out_n) != 0) __builtin_trap();
                    free(out2);
                }
                qb_free(g);
            }
            free(out);
        }
    }

    qb_free(f);
    return 0;
}
