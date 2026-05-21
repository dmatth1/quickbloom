// qb_util.c -- helpers exported from libquickbloom.

#include "quickbloom.h"
#include <assert.h>
#include <math.h>

size_t qb_estimate_bits(size_t n, double fp) {
    // Smallest meaningful filter is one 256-bit block.
    if (n == 0) return 256;
    // Catch the programmer bug in debug builds; release stays
    // conservative (return n*32) so production callers don't crash
    // on a stray sentinel value coming out of a config parser.
    assert(fp > 0.0 && fp < 1.0
           && "qb_estimate_bits: fp must be strictly between 0 and 1");
    if (!(fp > 0.0) || !(fp < 1.0)) return n * 32;

    // Classical Bloom filter bit-budget:
    //   m = -n * ln(fp) / (ln 2)^2
    // For SBBF (Parquet K=8) this is an approximation: SBBF's actual
    // FP rate at this sizing is typically within ~2x of fp on small
    // filters and converges to fp as the filter grows. Callers who
    // need a specific FP target on a small filter should over-size
    // (e.g. multiply the result by 1.5).
    const double LN2_SQ = 0.4804530139182014;  // ln(2)^2
    double m = -((double)n) * log(fp) / LN2_SQ;
    if (m < 256.0) m = 256.0;
    return (size_t)(m + 0.5);
}
