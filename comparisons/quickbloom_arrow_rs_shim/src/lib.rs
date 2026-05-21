// quickbloom_arrow_rs_shim/src/lib.rs
//
// 1:1 Rust port of quickbloom's SBBF (`bloom_sbbf.c`) for direct
// head-to-head against arrow-rs `parquet::bloom_filter::Sbbf` in
// the same bench harness.
//
// Faithful to quickbloom's optimization tricks:
//
//   - 256-bit blocks, K=8 bits on 32-bit lane boundaries, Parquet
//     SBBF SALT constants
//   - Power-of-2 block count + bitmask block-index (no fastrange
//     div; matches quickbloom's `((h >> 32) & nblocks_mask)`)
//   - 32-byte-aligned heap allocation (via std::alloc::Layout) so
//     `_mm256_load_si256` / `_mm256_store_si256` are aligned the
//     same way quickbloom's `posix_memalign(.., 32, ..)` does
//   - SIMD mask compute: vpmulld + vpsrld(27) + vpsllvd
//   - SIMD reduce: vpandn + vptest (via `_mm256_testc_si256`)
//   - 4-way unrolled bulk paths (mirrors `APPLY` macro + 4-way
//     loop in `bloom_insert_bulk` / `bloom_contains_bulk`)
//   - Software prefetch lookahead in bulk paths (PREFETCH_LOOKAHEAD
//     = 8, matching the `unified` variant)
//   - Scalar fallback for non-AVX2 builds
//
// Hash: XXH64 (matches arrow-rs and the Parquet spec) for an
// apples-to-apples comparison on the bytes-in path. The SBBF
// algorithm and SIMD tricks are 1:1 with quickbloom; the hash
// choice diverges from quickbloom's wymum to match arrow-rs's
// Sbbf so the bench's `miss`/`hit` numbers isolate the
// algorithm+SIMD delta from any hash differences.
//
// K_HASHES 8

#![allow(clippy::missing_safety_doc)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::os::raw::{c_int, c_void};
use std::slice;
use twox_hash::XxHash64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Parquet SBBF SALT constants, verbatim from the spec.
const SALT: [u32; 8] = [
    0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d,
    0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
];

const BLOCK_BYTES: usize = 32;        // 256-bit block
const BLOCK_LANES: usize = 8;         // 8 × u32 per block
const PREFETCH_LOOKAHEAD: usize = 8;  // matches quickbloom unified variant

// 32-byte-aligned heap buffer of u32, owning. Equivalent to
// quickbloom's posix_memalign-backed `bits` array.
struct AlignedU32Buf {
    ptr: *mut u32,
    len: usize,              // length in u32 elements
    layout: Layout,
}

unsafe impl Send for AlignedU32Buf {}
unsafe impl Sync for AlignedU32Buf {}

impl AlignedU32Buf {
    fn new_zeroed(len: usize) -> Self {
        let layout = Layout::from_size_align(len * 4, 32)
            .expect("u32 array layout with 32-byte alignment");
        // alloc_zeroed gives us zero-initialised memory — same as
        // quickbloom's posix_memalign + memset(0).
        let ptr = unsafe { alloc_zeroed(layout) } as *mut u32;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        AlignedU32Buf { ptr, len, layout }
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const u32 { self.ptr as *const u32 }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u32 { self.ptr }

    #[inline(always)]
    unsafe fn block_ptr(&self, base: usize) -> *const u32 {
        self.ptr.add(base) as *const u32
    }
    #[inline(always)]
    unsafe fn block_mut_ptr(&mut self, base: usize) -> *mut u32 {
        self.ptr.add(base)
    }
}

impl Drop for AlignedU32Buf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr as *mut u8, self.layout); }
    }
}

#[repr(C)]
struct Bloom {
    bits: AlignedU32Buf,         // length = num_blocks * 8
    num_blocks: usize,
    block_mask: u64,             // num_blocks - 1; num_blocks is power of two
}

impl Bloom {
    fn new(num_bytes: usize) -> Self {
        let nb_raw = (num_bytes.max(BLOCK_BYTES) + BLOCK_BYTES - 1) / BLOCK_BYTES;
        let nb = nb_raw.next_power_of_two();
        Bloom {
            bits: AlignedU32Buf::new_zeroed(nb * BLOCK_LANES),
            num_blocks: nb,
            block_mask: (nb - 1) as u64,
        }
    }

    #[inline(always)]
    fn block_base(&self, hash: u64) -> usize {
        // Quickbloom's block_for: ((h64 >> 32) & nblocks_mask) * 8.
        let idx = (hash >> 32) & self.block_mask;
        (idx as usize) * BLOCK_LANES
    }
}

// ---- mask_for: AVX2 K=8 mask vector (quickbloom mask_for) ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mask_for(h32: u32) -> __m256i {
    let salt = _mm256_loadu_si256(SALT.as_ptr() as *const __m256i);
    let hbcast = _mm256_set1_epi32(h32 as i32);
    let prod = _mm256_mullo_epi32(hbcast, salt);
    let shift = _mm256_srli_epi32::<27>(prod);
    let ones = _mm256_set1_epi32(1);
    _mm256_sllv_epi32(ones, shift)
}

// ---- Single-key AVX2 (mirrors QB_API(insert) / QB_API(contains)) ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn insert_hash_avx2(b: &mut Bloom, hash: u64) {
    let base = b.block_base(hash);
    let m = mask_for(hash as u32);
    let blk_ptr = b.bits.block_mut_ptr(base) as *mut __m256i;
    let cur = _mm256_load_si256(blk_ptr);
    _mm256_store_si256(blk_ptr, _mm256_or_si256(cur, m));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_hash_avx2(b: &Bloom, hash: u64) -> bool {
    let base = b.block_base(hash);
    let m = mask_for(hash as u32);
    let blk_ptr = b.bits.block_ptr(base) as *const __m256i;
    let cur = _mm256_load_si256(blk_ptr);
    _mm256_testc_si256(cur, m) != 0
}

// ---- Scalar fallback ----

fn insert_hash_scalar(b: &mut Bloom, hash: u64) {
    let base = b.block_base(hash);
    let key = hash as u32;
    unsafe {
        let p = b.bits.block_mut_ptr(base);
        for i in 0..8 {
            let bit = key.wrapping_mul(SALT[i]) >> 27;
            *p.add(i) |= 1u32 << bit;
        }
    }
}

fn check_hash_scalar(b: &Bloom, hash: u64) -> bool {
    let base = b.block_base(hash);
    let key = hash as u32;
    unsafe {
        let p = b.bits.block_ptr(base);
        for i in 0..8 {
            let bit = key.wrapping_mul(SALT[i]) >> 27;
            if (*p.add(i) >> bit) & 1 == 0 {
                return false;
            }
        }
    }
    true
}

#[inline(always)]
fn insert_hash(b: &mut Bloom, hash: u64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { insert_hash_avx2(b, hash); return; }
        }
    }
    insert_hash_scalar(b, hash);
}

#[inline(always)]
fn check_hash(b: &Bloom, hash: u64) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return check_hash_avx2(b, hash); }
        }
    }
    check_hash_scalar(b, hash)
}

// ---- 4-way bulk paths with prefetch lookahead ----
// Mirrors the APPLY macro + 4-way loop in quickbloom's
// bloom_insert_prehash_bulk / bloom_contains_prehash_bulk plus the
// PREFETCH_LOOKAHEAD prefetch from the `unified` variant.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn insert_prehash_bulk_avx2(b: &mut Bloom, hashes: &[u64]) {
    let n = hashes.len();
    let mut i = 0usize;
    let base_ptr = b.bits.as_mut_ptr();
    let block_mask = b.block_mask;
    while i + 4 <= n {
        // Prefetch lookahead — mirror quickbloom unified variant.
        if i + PREFETCH_LOOKAHEAD + 4 <= n {
            let pf0 = ((hashes[i + PREFETCH_LOOKAHEAD + 0] >> 32) & block_mask) as usize * BLOCK_LANES;
            let pf1 = ((hashes[i + PREFETCH_LOOKAHEAD + 1] >> 32) & block_mask) as usize * BLOCK_LANES;
            let pf2 = ((hashes[i + PREFETCH_LOOKAHEAD + 2] >> 32) & block_mask) as usize * BLOCK_LANES;
            let pf3 = ((hashes[i + PREFETCH_LOOKAHEAD + 3] >> 32) & block_mask) as usize * BLOCK_LANES;
            _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(pf0) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(pf1) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(pf2) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(pf3) as *const i8);
        }
        let h0 = hashes[i + 0];
        let h1 = hashes[i + 1];
        let h2 = hashes[i + 2];
        let h3 = hashes[i + 3];
        let b0 = ((h0 >> 32) & block_mask) as usize * BLOCK_LANES;
        let b1 = ((h1 >> 32) & block_mask) as usize * BLOCK_LANES;
        let b2 = ((h2 >> 32) & block_mask) as usize * BLOCK_LANES;
        let b3 = ((h3 >> 32) & block_mask) as usize * BLOCK_LANES;
        let m0 = mask_for(h0 as u32);
        let m1 = mask_for(h1 as u32);
        let m2 = mask_for(h2 as u32);
        let m3 = mask_for(h3 as u32);
        // APPLY: load+or+store, aligned, on each of the 4 blocks.
        let p0 = base_ptr.add(b0) as *mut __m256i;
        let p1 = base_ptr.add(b1) as *mut __m256i;
        let p2 = base_ptr.add(b2) as *mut __m256i;
        let p3 = base_ptr.add(b3) as *mut __m256i;
        _mm256_store_si256(p0, _mm256_or_si256(_mm256_load_si256(p0), m0));
        _mm256_store_si256(p1, _mm256_or_si256(_mm256_load_si256(p1), m1));
        _mm256_store_si256(p2, _mm256_or_si256(_mm256_load_si256(p2), m2));
        _mm256_store_si256(p3, _mm256_or_si256(_mm256_load_si256(p3), m3));
        i += 4;
    }
    while i < n {
        insert_hash_avx2(b, hashes[i]);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn contains_prehash_bulk_avx2(b: &Bloom, hashes: &[u64]) -> usize {
    let n = hashes.len();
    let mut i = 0usize;
    let mut hits = 0usize;
    let base_ptr = b.bits.as_ptr();
    let block_mask = b.block_mask;
    while i + 4 <= n {
        if i + PREFETCH_LOOKAHEAD + 4 <= n {
            let pf0 = ((hashes[i + PREFETCH_LOOKAHEAD + 0] >> 32) & block_mask) as usize * BLOCK_LANES;
            let pf1 = ((hashes[i + PREFETCH_LOOKAHEAD + 1] >> 32) & block_mask) as usize * BLOCK_LANES;
            let pf2 = ((hashes[i + PREFETCH_LOOKAHEAD + 2] >> 32) & block_mask) as usize * BLOCK_LANES;
            let pf3 = ((hashes[i + PREFETCH_LOOKAHEAD + 3] >> 32) & block_mask) as usize * BLOCK_LANES;
            _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(pf0) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(pf1) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(pf2) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(pf3) as *const i8);
        }
        let h0 = hashes[i + 0];
        let h1 = hashes[i + 1];
        let h2 = hashes[i + 2];
        let h3 = hashes[i + 3];
        let b0 = ((h0 >> 32) & block_mask) as usize * BLOCK_LANES;
        let b1 = ((h1 >> 32) & block_mask) as usize * BLOCK_LANES;
        let b2 = ((h2 >> 32) & block_mask) as usize * BLOCK_LANES;
        let b3 = ((h3 >> 32) & block_mask) as usize * BLOCK_LANES;
        let m0 = mask_for(h0 as u32);
        let m1 = mask_for(h1 as u32);
        let m2 = mask_for(h2 as u32);
        let m3 = mask_for(h3 as u32);
        let c0 = _mm256_load_si256(base_ptr.add(b0) as *const __m256i);
        let c1 = _mm256_load_si256(base_ptr.add(b1) as *const __m256i);
        let c2 = _mm256_load_si256(base_ptr.add(b2) as *const __m256i);
        let c3 = _mm256_load_si256(base_ptr.add(b3) as *const __m256i);
        hits += _mm256_testc_si256(c0, m0) as usize;
        hits += _mm256_testc_si256(c1, m1) as usize;
        hits += _mm256_testc_si256(c2, m2) as usize;
        hits += _mm256_testc_si256(c3, m3) as usize;
        i += 4;
    }
    while i < n {
        if check_hash_avx2(b, hashes[i]) { hits += 1; }
        i += 1;
    }
    hits
}

// XXH64 with seed 0 (Parquet spec; matches arrow-rs Sbbf).
//
// We use the one-shot API (twox-hash 2.x `oneshot`) -- this is the
// exact call shape arrow-rs's Sbbf uses internally, and it's ~5x
// faster than the streaming Hasher API (with_seed/write/finish)
// because it skips state-tracking and lets the optimizer inline the
// whole hash inline.
#[inline(always)]
fn xxh64(bytes: &[u8]) -> u64 {
    XxHash64::oneshot(0, bytes)
}

// Bytes-in bulk with prefetch on the hash-then-probe path. Mirrors
// quickbloom's bloom_insert_bulk / bloom_contains_bulk which do
// 4-way + prefetch when klen == 16. We accept any klen and use
// XXH64 (vs quickbloom's hash16 wymum fast-path) so it's apples-
// to-apples vs arrow-rs.

unsafe fn insert_bulk_bytes(b: &mut Bloom, keys: *const u8, klen: usize, n: usize) {
    // Hash all keys first into a small staging array of 4 at a time,
    // then dispatch to the AVX2 4-way insert. Same shape as quickbloom
    // when klen==16, here generalised to any klen.
    let mut hashes = [0u64; 4];
    let mut i = 0usize;
    while i + 4 <= n {
        for j in 0..4 {
            let key = slice::from_raw_parts(keys.add((i + j) * klen), klen);
            hashes[j] = xxh64(key);
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                insert_prehash_bulk_avx2(b, &hashes);
                i += 4;
                continue;
            }
        }
        for &h in hashes.iter() { insert_hash(b, h); }
        i += 4;
    }
    while i < n {
        let key = slice::from_raw_parts(keys.add(i * klen), klen);
        insert_hash(b, xxh64(key));
        i += 1;
    }
}

unsafe fn contains_bulk_bytes(b: &Bloom, keys: *const u8, klen: usize, n: usize) -> usize {
    let mut hashes = [0u64; 4];
    let mut i = 0usize;
    let mut hits = 0usize;
    while i + 4 <= n {
        for j in 0..4 {
            let key = slice::from_raw_parts(keys.add((i + j) * klen), klen);
            hashes[j] = xxh64(key);
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                hits += contains_prehash_bulk_avx2(b, &hashes);
                i += 4;
                continue;
            }
        }
        for &h in hashes.iter() {
            if check_hash(b, h) { hits += 1; }
        }
        i += 4;
    }
    while i < n {
        let key = slice::from_raw_parts(keys.add(i * klen), klen);
        if check_hash(b, xxh64(key)) { hits += 1; }
        i += 1;
    }
    hits
}

// ---- C ABI matching the rest of the bench harness ----

#[unsafe(no_mangle)]
pub extern "C" fn bloom_k_hashes() -> u32 { 8 }

#[unsafe(no_mangle)]
pub extern "C" fn bloom_new(nbits: usize) -> *mut c_void {
    let num_bytes = (nbits + 7) / 8;
    let f = Bloom::new(num_bytes);
    Box::into_raw(Box::new(f)) as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "C" fn bloom_free(p: *mut c_void) {
    if p.is_null() { return; }
    unsafe { drop(Box::from_raw(p as *mut Bloom)); }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert(p: *mut c_void, key: *const u8, len: usize) {
    let f = &mut *(p as *mut Bloom);
    let bytes = slice::from_raw_parts(key, len);
    insert_hash(f, xxh64(bytes));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains(p: *mut c_void, key: *const u8, len: usize) -> c_int {
    let f = &*(p as *const Bloom);
    let bytes = slice::from_raw_parts(key, len);
    if check_hash(f, xxh64(bytes)) { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert_bulk(
    p: *mut c_void, keys: *const u8, klen: usize, n: usize,
) {
    let f = &mut *(p as *mut Bloom);
    insert_bulk_bytes(f, keys, klen, n);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains_bulk(
    p: *mut c_void, keys: *const u8, klen: usize, n: usize,
) -> usize {
    let f = &*(p as *const Bloom);
    contains_bulk_bytes(f, keys, klen, n)
}

// Prehash API: lets the bench measure algorithm-only without the
// XXH64 cost. Arrow-rs Sbbf doesn't expose this (their API only
// takes AsBytes), so the harness shows (no prehash) for arrow_rs.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert_prehash(p: *mut c_void, hash: u64) {
    let f = &mut *(p as *mut Bloom);
    insert_hash(f, hash);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains_prehash(p: *mut c_void, hash: u64) -> c_int {
    let f = &*(p as *const Bloom);
    if check_hash(f, hash) { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert_prehash_bulk(
    p: *mut c_void, hashes: *const u64, n: usize,
) {
    let f = &mut *(p as *mut Bloom);
    let hs = slice::from_raw_parts(hashes, n);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            insert_prehash_bulk_avx2(f, hs);
            return;
        }
    }
    for &h in hs { insert_hash(f, h); }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains_prehash_bulk(
    p: *mut c_void, hashes: *const u64, n: usize,
) -> usize {
    let f = &*(p as *const Bloom);
    let hs = slice::from_raw_parts(hashes, n);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return contains_prehash_bulk_avx2(f, hs);
        }
    }
    let mut hits = 0usize;
    for &h in hs { if check_hash(f, h) { hits += 1; } }
    hits
}

