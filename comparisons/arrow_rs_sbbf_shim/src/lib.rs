// arrow_rs_sbbf_shim/src/lib.rs
//
// Shim exposing the bloom_* C ABI on top of arrow-rs
// (apache/arrow-rs) `parquet::bloom_filter::Sbbf`. This is the
// production-grade SBBF used by every Rust Parquet reader/writer.
//
// Sbbf uses XXH64 (the Parquet spec hash) internally and does not
// expose a prehash API, so bloom_*_prehash* are omitted; the harness
// will gracefully skip those benches for this candidate.
//
// K_HASHES 8

use parquet::bloom_filter::Sbbf;
use std::os::raw::{c_int, c_void};
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn bloom_k_hashes() -> u32 {
    8
}

#[unsafe(no_mangle)]
pub extern "C" fn bloom_new(nbits: usize) -> *mut c_void {
    // Sbbf is sized in bytes, not bits. Round up.
    let num_bytes = (nbits + 7) / 8;
    let filter = Sbbf::new_with_num_of_bytes(num_bytes);
    let boxed = Box::new(filter);
    Box::into_raw(boxed) as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "C" fn bloom_free(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(p as *mut Sbbf));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_insert(p: *mut c_void, key: *const u8, len: usize) {
    let f = &mut *(p as *mut Sbbf);
    let bytes = slice::from_raw_parts(key, len);
    f.insert(bytes);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_contains(p: *mut c_void, key: *const u8, len: usize) -> c_int {
    let f = &*(p as *const Sbbf);
    let bytes = slice::from_raw_parts(key, len);
    if f.check(bytes) {
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
    let f = &mut *(p as *mut Sbbf);
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
    let f = &*(p as *const Sbbf);
    let mut hits = 0usize;
    for i in 0..n {
        let bytes = slice::from_raw_parts(keys.add(i * klen), klen);
        if f.check(bytes) {
            hits += 1;
        }
    }
    hits
}

// No prehash API exposed: Sbbf::{insert,check} only take AsBytes
// values and always run XXH64 internally. If a future arrow-rs adds an
// insert_hash / check_hash, wire it up here.
