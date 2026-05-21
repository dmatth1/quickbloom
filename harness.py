"""Compile, correctness-test, and benchmark Bloom filter candidates.

Methodology (see also the Methodology section of README.md):

* Single-key API. We measure bulk_insert / bulk_contains, but those iterate
  over individual keys internally -- this is single-key throughput, not
  batched. Different from Krassovsky's batched 8-at-a-time numbers.
* Hash + bloom (default). The timed loop includes the candidate's own
  hash function. This is what a real user pays per insert/lookup.
* Pre-hashed bench mode (optional). For algorithm-only comparison vs
  published cyc/op figures, candidates can implement the prehash ABI
  (bloom_insert_prehash_bulk / bloom_contains_prehash_bulk). The harness
  uses random uint64 hash inputs and times only the bloom kernel.
* Multiple cache regimes. We sweep filter sizes from S (in-L2) to XL
  (out-of-L3) so the regime is always reported alongside the number.
* Statistics. Each phase runs `repeats` times after `warmup`; we report
  min, median, and p90 across the runs.

CFLAGS use the AVX2 baseline (-mavx2 -mbmi2 -mfma) -- no -march=native --
so .so is portable to any Haswell+ Intel and Zen2+ AMD.
"""
import ctypes
import hashlib
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import os

# Default compiler: respect CC env var, else the system cc. Set
# CC=clang on Sapphire Rapids+ for a few percent on the bloom kernel;
# the bench header records which compiler was used.
CC = os.environ.get("CC", "cc")

CFLAGS = [
    "-O3",
    # Portable AVX2 + AES-NI baseline. Available on:
    #   Intel: Westmere/Sandy Bridge (2010-2011) and later
    #   AMD:   Bulldozer (2011) and later
    # AES-NI is universal on modern x86_64 -- adding it costs no portability.
    "-mavx2", "-mbmi2", "-mfma", "-maes",
    "-fPIC", "-shared",
    # libm — needed by comparisons/bloom_xorfuse.c (binary fuse filter
    # uses log/round in size calculation). No-op for candidates that
    # don't call libm functions.
    "-lm",
]

# Filter-size presets are computed from the host's L2-per-core and
# L3 sizes so the four regimes (S=in-L2, M=just past L2, L=around L3,
# XL=DRAM) actually exercise those regimes on whatever box the bench
# runs on. Override the detection with QB_L2_KB / QB_L3_MB env vars
# for reproducibility across hosts.
#
# n_insert targets ~21 bits/key (SBBF natural load for K=8) so the
# filter is realistically populated; an under-loaded filter ages most
# blocks empty, which lets contains hit the early-exit path on the
# first bit and reports unrealistically low latencies. XL is
# intentionally quarter-loaded to keep peak harness memory in check.

def _read_cache_bytes(index: int) -> int:
    """Per-cpu cache size at /sys/devices/system/cpu/cpu0/cache/indexN/size.

    index2 is L2, index3 is L3. The kernel reports the cache *visible
    to cpu0*, which on every modern Intel/AMD part is the per-core L2
    and the shared L3. Returns 0 if the file is missing or unparseable.
    """
    try:
        with open(f"/sys/devices/system/cpu/cpu0/cache/index{index}/size") as f:
            v = f.read().strip()
    except OSError:
        return 0
    try:
        if v.endswith("K"): return int(v[:-1]) * 1024
        if v.endswith("M"): return int(v[:-1]) * 1024 * 1024
        return int(v)
    except ValueError:
        return 0


def detect_cache_bytes() -> tuple[int, int]:
    """Return (L2 per core, L3 total) in bytes for the host.

    Reads /sys first (most reliable); falls back to lscpu output and
    finally to a 1 MB L2 / 32 MB L3 default if neither is available.
    QB_L2_KB and QB_L3_MB env vars override the detected values.
    """
    env_l2 = os.environ.get("QB_L2_KB")
    env_l3 = os.environ.get("QB_L3_MB")

    l2 = int(env_l2) * 1024 if env_l2 else _read_cache_bytes(2)
    l3 = int(env_l3) * 1024 * 1024 if env_l3 else _read_cache_bytes(3)

    if not l2 or not l3:
        info = cpu_info()
        if not l2 and info["L2_kb"]:
            l2 = info["L2_kb"] * 1024
            # lscpu's L2 is sometimes aggregate across cores. Divide by
            # core count if it's implausibly large for a single core.
            n = os.cpu_count() or 1
            if l2 > 16 * 1024 * 1024 and n > 1:
                l2 //= n
        if not l3 and info["L3_mb"]:
            l3 = info["L3_mb"] * 1024 * 1024

    if not l2: l2 = 1 * 1024 * 1024
    if not l3: l3 = 32 * 1024 * 1024
    return l2, l3


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _prev_pow2(n: int) -> int:
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p


def compute_bench_sizes(l2_per_core: int, l3_total: int) -> dict:
    """Pick filter sizes that span the cache hierarchy on this host.

    Footprint targets:
      S  ~= L2 / 8           well inside L2 (round up; small fits anyway)
      M  ~= L2 * 2           just past per-core L2 (round up)
      L  ~= L3               sits right at L3 (round down so it fits)
      XL ~= L3 * 4           deep DRAM (round up)
    Capped so a single bench run stays under a few GB of memory on
    hosts with very large L3 (EPYC, Granite Rapids). L caps at 256 MB
    filter, XL at 512 MB filter — past that, you're measuring DRAM
    latency either way and bigger doesn't change the regime.
    n_insert is 21 bits/key for S/M/L (SBBF natural load); XL is
    quarter-loaded (84 bits/key) so the keys buffer + filter stay
    tractable.
    """
    # Filter-size caps (in bits) to bound per-run memory on hosts with
    # very large L3. L peaks at 256 MB filter (~102M keys × 16 B ≈
    # 1.6 GB keys buffer); XL peaks at 512 MB filter (~51M keys at
    # 84 bpk ≈ 820 MB keys buffer). Past those, the regime is
    # DRAM-bound regardless and bigger doesn't measure anything new.
    L_BITS_MAX  = 1 << 31
    XL_BITS_MAX = 1 << 32

    s_bits  = _next_pow2(max(l2_per_core // 8, 128 * 1024) * 8)
    m_bits  = _next_pow2(max(l2_per_core * 2,  2 * 1024 * 1024) * 8)
    l_bits  = min(max(_prev_pow2(l3_total * 8), m_bits * 4), L_BITS_MAX)
    xl_bits = min(_next_pow2(max(l3_total * 4, 512 * 1024 * 1024) * 8), XL_BITS_MAX)

    return {
        "S":  dict(nbits=s_bits,  n_insert=max(1, s_bits  // 21), n_query=  200_000),
        "M":  dict(nbits=m_bits,  n_insert=max(1, m_bits  // 21), n_query=  500_000),
        "L":  dict(nbits=l_bits,  n_insert=max(1, l_bits  // 21), n_query=  500_000),
        "XL": dict(nbits=xl_bits, n_insert=max(1, xl_bits // 84), n_query=2_000_000),
    }


L2_BYTES, L3_BYTES = detect_cache_bytes()
BENCH_SIZES = compute_bench_sizes(L2_BYTES, L3_BYTES)

KLEN_DEFAULT = 16
FP_RATE_LIMIT = 0.005

# Back-compat: the old constants several scripts still reference.
NBITS_DEFAULT    = BENCH_SIZES["S"]["nbits"]
N_INSERT_DEFAULT = BENCH_SIZES["S"]["n_insert"]
N_QUERY_DEFAULT  = BENCH_SIZES["S"]["n_query"]
NBITS_LARGE      = BENCH_SIZES["XL"]["nbits"]
N_INSERT_LARGE   = BENCH_SIZES["XL"]["n_insert"]
N_QUERY_LARGE    = BENCH_SIZES["XL"]["n_query"]


@dataclass
class Candidate:
    source_path: Path
    so_path: Path
    lib: ctypes.CDLL
    k_hashes: int
    has_prehash: bool
    has_batch8: bool


def parse_k_hashes(source: str) -> int:
    # Match either C's `#define K_HASHES 8` or a Rust comment marker
    # `// K_HASHES 8` so Rust shim crates (which have no #define) can
    # declare their K via a comment that the harness can read.
    m = re.search(r"K_HASHES\s+(\d+)", source)
    return int(m.group(1)) if m else 7


CFLAGS_AVX512 = ["-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl"]


# Filenames whose .c file produces qb_* symbols (the quickbloom
# library). Comparison candidates in comparisons/ keep the legacy
# bloom_* symbol names.
QB_CANDIDATES = {"quickbloom"}


def _alias_qb_to_bloom(lib: ctypes.CDLL) -> None:
    """Quickbloom exports qb_* symbols; the rest of the harness still
    addresses functions by their legacy `bloom_*` names. Alias on the
    Python wrapper so downstream harness code keeps working."""
    names = [
        "new", "free", "insert", "contains",
        "insert_bulk", "contains_bulk",
        "insert_prehash", "contains_prehash",
        "insert_prehash_bulk", "contains_prehash_bulk",
    ]
    for fname in names:
        try:
            fn = getattr(lib, f"qb_{fname}")
        except AttributeError:
            continue
        setattr(lib, f"bloom_{fname}", fn)


class CargoMissingError(RuntimeError):
    """Raised when a Rust crate candidate can't be built because the
    Rust toolchain (cargo) is not installed. Callers can catch this to
    skip Rust candidates gracefully on hosts without rustc."""


def _build_rust_crate(crate_dir: Path) -> Path:
    """Run `cargo build --release` for a cdylib crate at crate_dir and
    return the path to the produced .so. Crate name must match the
    directory name."""
    try:
        r = subprocess.run(
            ["cargo", "build", "--release",
             "--manifest-path", str(crate_dir / "Cargo.toml")],
            capture_output=True, text=True,
        )
    except FileNotFoundError as e:
        raise CargoMissingError(
            f"cargo not found on PATH; cannot build Rust candidate at {crate_dir}"
        ) from e
    if r.returncode != 0:
        raise RuntimeError(
            f"cargo build failed for {crate_dir}:\n{r.stderr or r.stdout}"
        )
    so = crate_dir / "target" / "release" / f"lib{crate_dir.name}.so"
    if not so.exists():
        raise RuntimeError(
            f"cargo built {crate_dir} but expected output {so} is missing"
        )
    return so


def compile_candidate(c_path: Path, out_dir: Path) -> Candidate:
    """Build a candidate and return the loaded library with symbol bindings.

    Accepts either:
      - a `.c` file (built with cc + standard CFLAGS), or
      - a directory containing a `Cargo.toml` (built with `cargo
        build --release`; cdylib crate; the produced
        `lib<dirname>.so` is loaded).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if c_path.is_dir() and (c_path / "Cargo.toml").is_file():
        so = _build_rust_crate(c_path)
        source_path = c_path / "src" / "lib.rs"
        is_qb = False
    else:
        so = out_dir / (c_path.stem + ".so")
        # Files with "_avx512" in their stem get the extra AVX-512 flags; the
        # resulting .so won't load on AVX2-only hardware but is fine for the
        # specific experiment of testing AVX-512 candidates.
        extra = CFLAGS_AVX512 if "_avx512" in c_path.stem else []
        is_qb = c_path.stem in QB_CANDIDATES
        r = subprocess.run(
            [CC, *CFLAGS, *extra, str(c_path), "-o", str(so)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"compile failed:\n{r.stderr}")
        source_path = c_path

    lib = ctypes.CDLL(str(so))
    if is_qb:
        _alias_qb_to_bloom(lib)
    lib.bloom_new.argtypes = [ctypes.c_size_t]
    lib.bloom_new.restype = ctypes.c_void_p
    lib.bloom_free.argtypes = [ctypes.c_void_p]
    lib.bloom_free.restype = None
    lib.bloom_insert.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.bloom_insert.restype = None
    lib.bloom_contains.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.bloom_contains.restype = ctypes.c_int
    lib.bloom_insert_bulk.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.bloom_insert_bulk.restype = None
    lib.bloom_contains_bulk.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t,
    ]
    lib.bloom_contains_bulk.restype = ctypes.c_size_t

    has_prehash = True
    try:
        lib.bloom_insert_prehash_bulk.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ]
        lib.bloom_insert_prehash_bulk.restype = None
        lib.bloom_contains_prehash_bulk.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ]
        lib.bloom_contains_prehash_bulk.restype = ctypes.c_size_t
    except AttributeError:
        has_prehash = False

    has_batch8 = True
    try:
        # Single-batch entry points (Krassovsky-compatible signatures).
        lib.bloom_insert_batch8.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.bloom_insert_batch8.restype = None
        lib.bloom_contains_batch8.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.bloom_contains_batch8.restype = ctypes.c_uint8
        # Bulk wrappers (n must be multiple of 8; tail handled in C).
        lib.bloom_insert_batch8_bulk.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ]
        lib.bloom_insert_batch8_bulk.restype = None
        lib.bloom_contains_batch8_bulk.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ]
        lib.bloom_contains_batch8_bulk.restype = ctypes.c_size_t
    except AttributeError:
        has_batch8 = False

    source = source_path.read_text() if source_path.is_file() else ""
    # Rust shims expose K via an extern function; prefer that over parsing.
    k = 0
    try:
        lib.bloom_k_hashes.restype = ctypes.c_uint32
        k = int(lib.bloom_k_hashes())
    except AttributeError:
        pass
    if k <= 0:
        k = parse_k_hashes(source)
    return Candidate(
        source_path=source_path, so_path=so, lib=lib,
        k_hashes=k,
        has_prehash=has_prehash,
        has_batch8=has_batch8,
    )


def make_corpus(n: int, klen: int, seed: int) -> bytes:
    """Deterministic packed buffer of n*klen bytes."""
    out = bytearray(n * klen)
    base = seed.to_bytes(8, "little")
    pos = 0
    for i in range(n):
        h = hashlib.sha256(base + i.to_bytes(8, "little")).digest()
        out[pos:pos + klen] = h[:klen]
        pos += klen
    return bytes(out)


def make_hashes(n: int, seed: int) -> tuple[bytes, ctypes.Array]:
    """Deterministic random uint64 hashes (for prehash bench).

    Returns (raw_bytes, ctypes_array_view). Keep the bytes alive while
    using the ctypes view -- ctypes from_buffer aliases the storage.
    """
    raw = make_corpus(n, 8, seed)
    arr = (ctypes.c_uint64 * n).from_buffer_copy(raw)
    return raw, arr


# --- Diff test (same as before; insert via key path only) -----------------
@dataclass
class DiffResult:
    passed: bool
    false_negatives: int
    false_positives: int
    fp_rate: float
    theoretical_fp_rate: float
    reason: str = ""


def theoretical_fp(k: int, n: int, m: int) -> float:
    return (1.0 - math.exp(-k * n / m)) ** k


def diff_test(c: Candidate,
              nbits: int = NBITS_DEFAULT,
              n_insert: int = N_INSERT_DEFAULT,
              n_query: int = N_QUERY_DEFAULT,
              klen: int = KLEN_DEFAULT,
              fp_limit: float = FP_RATE_LIMIT) -> DiffResult:
    inserted = make_corpus(n_insert, klen, seed=1)
    unseen = make_corpus(n_query, klen, seed=2)
    bf = c.lib.bloom_new(nbits)
    try:
        c.lib.bloom_insert_bulk(bf, inserted, klen, n_insert)
        hits_in = c.lib.bloom_contains_bulk(bf, inserted, klen, n_insert)
        hits_out = c.lib.bloom_contains_bulk(bf, unseen, klen, n_query)
        fn = n_insert - hits_in
        fp = hits_out
        fp_rate = fp / n_query
        theo = theoretical_fp(c.k_hashes, n_insert, nbits)
        passed = True
        reason = ""
        if fn > 0:
            passed = False; reason = f"false negatives: {fn}"
        elif fp_rate > fp_limit:
            passed = False; reason = f"FP rate {fp_rate:.4f} > limit {fp_limit:.4f}"
        return DiffResult(passed, fn, fp, fp_rate, theo, reason)
    finally:
        c.lib.bloom_free(bf)


# --- Benchmark ------------------------------------------------------------
@dataclass
class PhaseStats:
    """Per-phase bench stats: list of ns/op samples + summary stats."""
    samples_ns: list[float] = field(default_factory=list)

    @property
    def min(self) -> float: return min(self.samples_ns) if self.samples_ns else float("nan")
    @property
    def median(self) -> float:
        return statistics.median(self.samples_ns) if self.samples_ns else float("nan")
    @property
    def p90(self) -> float:
        if len(self.samples_ns) < 2: return self.min
        # statistics.quantiles needs >= 2 samples
        try:
            qs = statistics.quantiles(self.samples_ns, n=10, method="inclusive")
            return qs[8]   # 90th percentile
        except statistics.StatisticsError:
            return self.min


@dataclass
class BenchResult:
    insert: PhaseStats = field(default_factory=PhaseStats)
    miss:   PhaseStats = field(default_factory=PhaseStats)
    hit:    PhaseStats = field(default_factory=PhaseStats)

    # Convenience: legacy access for `.insert_ns_per_op` etc. so old
    # scripts keep working (returns the min, which is what they used).
    @property
    def insert_ns_per_op(self) -> float: return self.insert.min
    @property
    def contains_miss_ns_per_op(self) -> float: return self.miss.min
    @property
    def contains_hit_ns_per_op(self) -> float: return self.hit.min


def benchmark(c: Candidate,
              nbits: int = NBITS_DEFAULT,
              n_insert: int = N_INSERT_DEFAULT,
              n_query: int = N_QUERY_DEFAULT,
              klen: int = KLEN_DEFAULT,
              repeats: int = 9, warmup: int = 2) -> BenchResult:
    """Hash-included bench: timed loop calls candidate's bulk ops on raw keys.

    This is what a real user pays per insert/lookup. To remove the hash
    contribution and compare against published bloom-kernel-only figures,
    use benchmark_prehash().
    """
    inserted = make_corpus(n_insert, klen, seed=42)
    unseen = make_corpus(n_query, klen, seed=43)
    res = BenchResult()
    for run in range(warmup + repeats):
        bf = c.lib.bloom_new(nbits)
        try:
            t0 = time.perf_counter_ns()
            c.lib.bloom_insert_bulk(bf, inserted, klen, n_insert)
            t_ins = (time.perf_counter_ns() - t0) / n_insert
            t0 = time.perf_counter_ns()
            c.lib.bloom_contains_bulk(bf, unseen, klen, n_query)
            t_miss = (time.perf_counter_ns() - t0) / n_query
            t0 = time.perf_counter_ns()
            c.lib.bloom_contains_bulk(bf, inserted, klen, n_insert)
            t_hit = (time.perf_counter_ns() - t0) / n_insert
            if run >= warmup:
                res.insert.samples_ns.append(t_ins)
                res.miss.samples_ns.append(t_miss)
                res.hit.samples_ns.append(t_hit)
        finally:
            c.lib.bloom_free(bf)
    return res


def benchmark_large(c: Candidate, **kw) -> BenchResult:
    """Back-compat: alias for the XL preset."""
    sz = BENCH_SIZES["XL"]
    return benchmark(c, nbits=sz["nbits"], n_insert=sz["n_insert"],
                     n_query=sz["n_query"], **kw)


def benchmark_at(c: Candidate, size: str, **kw) -> BenchResult:
    """Bench using a named preset (S/M/L/XL)."""
    sz = BENCH_SIZES[size]
    return benchmark(c, nbits=sz["nbits"], n_insert=sz["n_insert"],
                     n_query=sz["n_query"], **kw)


def benchmark_prehash(c: Candidate,
                      nbits: int = NBITS_DEFAULT,
                      n_insert: int = N_INSERT_DEFAULT,
                      n_query: int = N_QUERY_DEFAULT,
                      repeats: int = 9, warmup: int = 2) -> BenchResult | None:
    """Algorithm-only bench: hashes are pre-computed outside the timed loop.

    Apples-to-apples against published 'cycles per op' figures from
    Save Buffer, Krassovsky, FastFilter etc. that exclude hash from the
    timed window. Returns None if the candidate doesn't expose the
    prehash ABI.

    The pre-hashed inputs are uniformly-random uint64s -- bloom kernels
    only require uniform input, so this is a fair comparison even when
    different candidates would normally use different hash functions.
    """
    if not c.has_prehash:
        return None
    raw_in,  arr_in  = make_hashes(n_insert, seed=42)
    raw_un,  arr_un  = make_hashes(n_query,  seed=43)
    res = BenchResult()
    for run in range(warmup + repeats):
        bf = c.lib.bloom_new(nbits)
        try:
            t0 = time.perf_counter_ns()
            c.lib.bloom_insert_prehash_bulk(bf, arr_in, n_insert)
            t_ins = (time.perf_counter_ns() - t0) / n_insert
            t0 = time.perf_counter_ns()
            c.lib.bloom_contains_prehash_bulk(bf, arr_un, n_query)
            t_miss = (time.perf_counter_ns() - t0) / n_query
            t0 = time.perf_counter_ns()
            c.lib.bloom_contains_prehash_bulk(bf, arr_in, n_insert)
            t_hit = (time.perf_counter_ns() - t0) / n_insert
            if run >= warmup:
                res.insert.samples_ns.append(t_ins)
                res.miss.samples_ns.append(t_miss)
                res.hit.samples_ns.append(t_hit)
        finally:
            c.lib.bloom_free(bf)
    return res


def benchmark_prehash_at(c: Candidate, size: str, **kw):
    sz = BENCH_SIZES[size]
    return benchmark_prehash(c, nbits=sz["nbits"], n_insert=sz["n_insert"],
                             n_query=sz["n_query"], **kw)


def benchmark_batch8(c: Candidate,
                     nbits: int = NBITS_DEFAULT,
                     n_insert: int = N_INSERT_DEFAULT,
                     n_query: int = N_QUERY_DEFAULT,
                     repeats: int = 9, warmup: int = 2) -> BenchResult | None:
    """8-way batched bench: bloom_*_batch8_bulk on pre-computed hashes.

    Matches Krassovsky's published 2.5 cyc/op regime (PatternedSimdBloomFilter):
    8 hashes per call, gather-friendly. Returns None if the candidate
    doesn't expose the batch8 ABI.

    Numbers reported are per-key (total elapsed / n), so directly
    comparable to single-key and prehash benches.
    """
    if not c.has_batch8:
        return None
    # n_insert and n_query rounded down to multiples of 8.
    n_insert = (n_insert // 8) * 8
    n_query  = (n_query  // 8) * 8
    raw_in,  arr_in  = make_hashes(n_insert, seed=42)
    raw_un,  arr_un  = make_hashes(n_query,  seed=43)
    res = BenchResult()
    for run in range(warmup + repeats):
        bf = c.lib.bloom_new(nbits)
        try:
            t0 = time.perf_counter_ns()
            c.lib.bloom_insert_batch8_bulk(bf, arr_in, n_insert)
            t_ins = (time.perf_counter_ns() - t0) / n_insert
            t0 = time.perf_counter_ns()
            c.lib.bloom_contains_batch8_bulk(bf, arr_un, n_query)
            t_miss = (time.perf_counter_ns() - t0) / n_query
            t0 = time.perf_counter_ns()
            c.lib.bloom_contains_batch8_bulk(bf, arr_in, n_insert)
            t_hit = (time.perf_counter_ns() - t0) / n_insert
            if run >= warmup:
                res.insert.samples_ns.append(t_ins)
                res.miss.samples_ns.append(t_miss)
                res.hit.samples_ns.append(t_hit)
        finally:
            c.lib.bloom_free(bf)
    return res


def benchmark_batch8_at(c: Candidate, size: str, **kw):
    sz = BENCH_SIZES[size]
    return benchmark_batch8(c, nbits=sz["nbits"], n_insert=sz["n_insert"],
                            n_query=sz["n_query"], **kw)


# --- CPU info reporting ---------------------------------------------------
def cpu_info() -> dict:
    """Return CPU model, freq, cache sizes for the methodology header."""
    info = {"model": "unknown", "mhz": 0.0,
            "L1d_kb": 0, "L2_kb": 0, "L3_mb": 0}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and info["model"] == "unknown":
                    info["model"] = line.split(":", 1)[1].strip()
                elif line.startswith("cpu MHz") and info["mhz"] == 0:
                    info["mhz"] = float(line.split(":", 1)[1].strip())
    except Exception:
        pass
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("L1d cache:"):
                v = line.split(":", 1)[1].strip()
                if "KiB" in v: info["L1d_kb"] = int(v.split()[0])
            elif line.startswith("L2 cache:"):
                v = line.split(":", 1)[1].strip()
                if "MiB" in v: info["L2_kb"] = int(float(v.split()[0]) * 1024)
                elif "KiB" in v: info["L2_kb"] = int(v.split()[0])
            elif line.startswith("L3 cache:"):
                v = line.split(":", 1)[1].strip()
                if "MiB" in v: info["L3_mb"] = int(float(v.split()[0]))
                elif "GiB" in v: info["L3_mb"] = int(float(v.split()[0]) * 1024)
    except Exception:
        pass
    return info
