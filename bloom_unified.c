// bloom_unified.c -- SBBF + prefetch lookahead of 8 keys.
//
// Single-source build: this file selects PREFETCH_LOOKAHEAD=8 and pulls
// in the SBBF implementation from bloom_sbbf.c.
//
// Recommended as the default when the filter size is not known at
// construction time, or you want a single design that holds across the
// cache hierarchy. Slight in-cache regression vs single_key; clear wins
// at M, L, XL.
#define K_HASHES 8
#define PREFETCH_LOOKAHEAD 8
#include "bloom_sbbf.c"
