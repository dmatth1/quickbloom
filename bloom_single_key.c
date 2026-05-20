// bloom_single_key.c -- SBBF, no prefetch. Lowest in-cache latency.
//
// Entry point for the "single_key" variant. Selects PREFETCH_LOOKAHEAD=0
// and pulls in the shared SBBF implementation from bloom_sbbf.c. Public
// functions are exported as qb_single_key_* via the QB_NS macro.
//
// Recommended for: in-cache filters (S/M sizes; < ~10 MB) or when the
// filter size isn't known in advance and you want the lowest per-op
// in-cache latency.
#define QB_NS qb_single_key
#define K_HASHES 8
#define PREFETCH_LOOKAHEAD 0
#include "bloom_sbbf.c"
