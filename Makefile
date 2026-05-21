# quickbloom Makefile
#
# Targets:
#   make            -- build the static and shared libraries (default)
#   make lib        -- build build/libquickbloom.{a,so}
#   make test       -- build and run the native C test binary
#   make example    -- build the hello_quickbloom example
#   make bench      -- run the Python benchmark sweep
#   make install    -- install header + libs + pkg-config to $(PREFIX) (default /usr/local)
#   make clean      -- remove build artefacts
#
# Variables (override on the command line):
#   CC=clang        -- C compiler (default: cc)
#   PREFIX=...      -- install root (default: /usr/local)
#   DESTDIR=...     -- staging root prepended to install paths
#   CFLAGS_EXTRA=.. -- additional compiler flags

CC      ?= cc
PREFIX  ?= /usr/local
BUILD   := build

VERSION    := 0.1.0
SOVERSION  := 0

CFLAGS_BASE := -O3 -mavx2 -mbmi2 -mfma -maes -fPIC -Wall -Wextra -std=c11
CFLAGS      := $(CFLAGS_BASE) $(CFLAGS_EXTRA)

# Each variant compiles bloom_sbbf.c or bloom_batched.c with a different
# QB_NS and PREFETCH_LOOKAHEAD. qb_util.o has variant-agnostic helpers
# (qb_estimate_bits). All four objects link into the same library.
OBJS := \
	$(BUILD)/single_key.o \
	$(BUILD)/unified.o    \
	$(BUILD)/batched.o    \
	$(BUILD)/qb_util.o

# Linux-style versioned shared library:
#   libquickbloom.so.0.1.0  -- actual file
#   libquickbloom.so.0      -- SONAME (ABI generation; symlink to .0.1.0)
#   libquickbloom.so        -- linker name (symlink to .0)
LIB_SO_REAL := $(BUILD)/libquickbloom.so.$(VERSION)
LIB_SO_SO   := $(BUILD)/libquickbloom.so.$(SOVERSION)
LIB_SO      := $(BUILD)/libquickbloom.so
LIB_A       := $(BUILD)/libquickbloom.a
LIB_PC      := $(BUILD)/quickbloom.pc

.PHONY: all lib test example bench bench-hash bench-atomic install clean

all: lib

lib: $(LIB_SO) $(LIB_A) $(LIB_PC)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/single_key.o: bloom_sbbf.c bloom_single_key.c | $(BUILD)
	$(CC) $(CFLAGS) -DQB_NS=qb_single_key -DPREFETCH_LOOKAHEAD=0 -c bloom_sbbf.c -o $@

$(BUILD)/unified.o: bloom_sbbf.c bloom_unified.c | $(BUILD)
	$(CC) $(CFLAGS) -DQB_NS=qb_unified -DPREFETCH_LOOKAHEAD=8 -c bloom_sbbf.c -o $@

$(BUILD)/batched.o: bloom_batched.c | $(BUILD)
	$(CC) $(CFLAGS) -c bloom_batched.c -o $@

$(BUILD)/qb_util.o: qb_util.c quickbloom.h | $(BUILD)
	$(CC) $(CFLAGS) -I. -c qb_util.c -o $@

$(LIB_SO_REAL): $(OBJS)
	$(CC) -shared -Wl,-soname,libquickbloom.so.$(SOVERSION) -o $@ $(OBJS) -lm

$(LIB_SO): $(LIB_SO_REAL)
	cd $(BUILD) && ln -sf libquickbloom.so.$(VERSION)    libquickbloom.so.$(SOVERSION)
	cd $(BUILD) && ln -sf libquickbloom.so.$(SOVERSION)  libquickbloom.so

$(LIB_A): $(OBJS)
	ar rcs $@ $(OBJS)

$(LIB_PC): quickbloom.pc.in | $(BUILD)
	sed -e 's|@PREFIX@|$(PREFIX)|g' \
	    -e 's|@VERSION@|$(VERSION)|g' \
	    quickbloom.pc.in > $@

# Native C tests. Links against the static library so a system without
# the shared lib installed can still run them. -lm for qb_estimate_bits.
$(BUILD)/test_quickbloom: test/test_quickbloom.c $(LIB_A) quickbloom.h | $(BUILD)
	$(CC) $(CFLAGS) -I. -o $@ test/test_quickbloom.c $(LIB_A) -lm

test: $(BUILD)/test_quickbloom
	$<

# Example. Same linkage shape as a consumer would use.
$(BUILD)/hello_quickbloom: examples/hello_quickbloom.c $(LIB_A) quickbloom.h | $(BUILD)
	$(CC) $(CFLAGS) -I. -o $@ examples/hello_quickbloom.c $(LIB_A) -lm

example: $(BUILD)/hello_quickbloom
	$<

# Python benchmark sweep. Compiles each variant's .c separately the way
# it always has — independent of the library build above.
bench:
	python3 bench_all.py

# Per-hash cost bench. Measures wymum16 / XXH64 / SipHash-1-3 over 16-byte
# keys with the same compile flags as the rest of the project, so the
# numbers can be compared directly against the prehash bloom benches.
$(BUILD)/bench_hash: tools/bench_hash.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tools/bench_hash.c

bench-hash: $(BUILD)/bench_hash
	$<

# Concurrent-insert prototype: measures atomic vs non-atomic block-OR.
# Quantifies the cost of making qb_*_insert safe under concurrent writers.
$(BUILD)/bench_atomic_insert: tools/bench_atomic_insert.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tools/bench_atomic_insert.c

bench-atomic: $(BUILD)/bench_atomic_insert
	$<

install: lib
	install -d $(DESTDIR)$(PREFIX)/include
	install -d $(DESTDIR)$(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 quickbloom.h    $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(LIB_A)        $(DESTDIR)$(PREFIX)/lib/
	install -m 755 $(LIB_SO_REAL)  $(DESTDIR)$(PREFIX)/lib/
	install -m 644 $(LIB_PC)       $(DESTDIR)$(PREFIX)/lib/pkgconfig/
	cd $(DESTDIR)$(PREFIX)/lib && ln -sf libquickbloom.so.$(VERSION)    libquickbloom.so.$(SOVERSION)
	cd $(DESTDIR)$(PREFIX)/lib && ln -sf libquickbloom.so.$(SOVERSION)  libquickbloom.so

clean:
	rm -rf $(BUILD)
