// bloom_single_key.c -- SBBF, no prefetch. Lowest in-cache latency.
//
// Single-source build: this file selects PREFETCH_LOOKAHEAD=0 and pulls
// in the SBBF implementation from bloom_sbbf.c.
//
// Recommended for: in-cache filters (S/M sizes; < ~10 MB) or when the
// filter size isn't known in advance and you want the lowest per-op
// in-cache latency.
#define K_HASHES 8
#define PREFETCH_LOOKAHEAD 0
#include "bloom_sbbf.c"
