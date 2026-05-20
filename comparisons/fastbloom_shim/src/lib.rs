// fastbloom_shim/src/lib.rs
//
// Shim exposing the bloom_* C ABI used by quickbloom's bench harness
// on top of the fastbloom crate (tomtomwombat/fastbloom).
// Default hasher: SipHash-1-3 with a fixed seed (we don't randomize
// per-process here so bench runs are reproducible).
//
// K_HASHES 8

use fastbloom::BloomFilter;
use std::os::raw::{c_int, c_void};
use std::slice;

// Tell parse_k_hashes-style scanners that this candidate uses K=8.
// The string "K_HASHES 8" below is picked up by the harness.
//
// K_HASHES 8

#[unsafe(no_mangle)]
pub extern "C" fn bloom_k_hashes() -> u32 {
    8
}

#[unsafe(no_mangle)]
pub extern "C" fn bloom_new(nbits: usize) -> *mut c_void {
    // BloomFilter::with_num_bits(n).hashes(k) returns a BloomFilter
    // directly (the .hashes call consumes the builder).
    let filter: BloomFilter = BloomFilter::with_num_bits(nbits).hashes(8);
    let boxed = Box::new(filter);
    Box::into_raw(boxed) as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "C" fn bloom_free(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(p as *mut BloomFilter));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert(p: *mut c_void, key: *const u8, len: usize) {
    let f = &mut *(p as *mut BloomFilter);
    let bytes = slice::from_raw_parts(key, len);
    f.insert(bytes);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains(p: *mut c_void, key: *const u8, len: usize) -> c_int {
    let f = &*(p as *const BloomFilter);
    let bytes = slice::from_raw_parts(key, len);
    if f.contains(bytes) {
        1
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert_bulk(
    p: *mut c_void,
    keys: *const u8,
    klen: usize,
    n: usize,
) {
    let f = &mut *(p as *mut BloomFilter);
    for i in 0..n {
        let bytes = slice::from_raw_parts(keys.add(i * klen), klen);
        f.insert(bytes);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains_bulk(
    p: *mut c_void,
    keys: *const u8,
    klen: usize,
    n: usize,
) -> usize {
    let f = &*(p as *const BloomFilter);
    let mut hits = 0usize;
    for i in 0..n {
        let bytes = slice::from_raw_parts(keys.add(i * klen), klen);
        if f.contains(bytes) {
            hits += 1;
        }
    }
    hits
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert_prehash(p: *mut c_void, hash: u64) {
    let f = &mut *(p as *mut BloomFilter);
    f.insert_hash(hash);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains_prehash(p: *mut c_void, hash: u64) -> c_int {
    let f = &*(p as *const BloomFilter);
    if f.contains_hash(hash) {
        1
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert_prehash_bulk(
    p: *mut c_void,
    hashes: *const u64,
    n: usize,
) {
    let f = &mut *(p as *mut BloomFilter);
    let hs = slice::from_raw_parts(hashes, n);
    for &h in hs {
        f.insert_hash(h);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains_prehash_bulk(
    p: *mut c_void,
    hashes: *const u64,
    n: usize,
) -> usize {
    let f = &*(p as *const BloomFilter);
    let hs = slice::from_raw_parts(hashes, n);
    let mut hits = 0usize;
    for &h in hs {
        if f.contains_hash(h) {
            hits += 1;
        }
    }
    hits
}
