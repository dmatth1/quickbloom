// hello_quickbloom.c -- minimal usage example.
//
// Build with the library installed at $PREFIX:
//   cc -O3 -mavx2 -mbmi2 -mfma -maes hello_quickbloom.c -lquickbloom -lm -o hello_quickbloom
//
// Or, if building inside this repo via the top-level Makefile, just
// run `make example` and it links against build/libquickbloom.a.

#include <quickbloom.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    // Size the filter for ~10,000 items at ~1% false-positive rate.
    void* f = qb_new(qb_estimate_bits(10000, 0.01));
    if (!f) {
        fprintf(stderr, "qb_new: out of memory\n");
        return 1;
    }

    const char* present[] = { "Love", "is", "in", "bloom" };
    const char* missing[] = { "absent", "missing", "ghost" };

    for (size_t i = 0; i < sizeof(present) / sizeof(present[0]); i++) {
        qb_insert(f, present[i], strlen(present[i]));
    }

    printf("present probes:\n");
    for (size_t i = 0; i < sizeof(present) / sizeof(present[0]); i++) {
        int hit = qb_contains(f, present[i], strlen(present[i]));
        printf("  %-8s => %s\n", present[i], hit ? "in (true positive)" : "MISS (BUG)");
    }

    printf("absent probes:\n");
    int fps = 0;
    for (size_t i = 0; i < sizeof(missing) / sizeof(missing[0]); i++) {
        int hit = qb_contains(f, missing[i], strlen(missing[i]));
        printf("  %-8s => %s\n", missing[i], hit ? "in (false positive)" : "out");
        if (hit) fps++;
    }
    printf("(false positives in this run: %d)\n", fps);

    qb_free(f);
    return 0;
}
