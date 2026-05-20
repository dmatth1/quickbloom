// bloom_unified.c -- SBBF + prefetch lookahead of 8 keys.
//
// Entry point for the "unified" variant. Selects PREFETCH_LOOKAHEAD=8
// and pulls in the shared SBBF implementation from bloom_sbbf.c. Public
// functions are exported as qb_unified_* via the QB_NS macro.
//
// Recommended as the default when the filter size is not known at
// construction time, or you want a single design that holds across the
// cache hierarchy. Slight in-cache regression vs single_key; clear wins
// at M, L, XL.
#define QB_NS qb_unified
#define K_HASHES 8
#define PREFETCH_LOOKAHEAD 8
#include "bloom_sbbf.c"
