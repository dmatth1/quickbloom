# quickbloom Makefile
#
# Targets:
#   make            -- build the static and shared libraries (default)
#   make lib        -- build build/libquickbloom.{a,so}
#   make test       -- build and run the native C test binary
#   make example    -- build the hello_quickbloom example
#   make bench      -- run the Python benchmark sweep
#   make bench-hash -- per-hash kernel cost (wymum / XXH64 / SipHash-1-3)
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

OBJS := \
	$(BUILD)/quickbloom.o \
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

.PHONY: all lib test example bench bench-hash fuzz install clean

all: lib

lib: $(LIB_SO) $(LIB_A) $(LIB_PC)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/quickbloom.o: quickbloom.c quickbloom.h | $(BUILD)
	$(CC) $(CFLAGS) -I. -c quickbloom.c -o $@

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

# Python benchmark sweep. Compiles each candidate's .c separately the
# way it always has — independent of the library build above.
bench:
	python3 bench_all.py

# Per-hash cost bench. Measures wymum16 / XXH64 / SipHash-1-3 over
# 16-byte keys with the same compile flags as the rest of the
# project, so the numbers can be compared directly against the
# prehash bloom benches.
$(BUILD)/bench_hash: tools/bench_hash.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tools/bench_hash.c

bench-hash: $(BUILD)/bench_hash
	$<

# libFuzzer harnesses. Require clang. Build the library with the
# fuzzer sanitizer so the bloom code is instrumented too, then link
# each harness into a standalone fuzzer binary.
FUZZ_CFLAGS := -O1 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
               -mavx2 -mbmi2 -mfma -maes -Wall -Wextra -std=c11

$(BUILD)/fuzz_insert: tools/fuzz_insert.c quickbloom.c qb_util.c quickbloom.h | $(BUILD)
	$(CC) $(FUZZ_CFLAGS) -I. -o $@ tools/fuzz_insert.c quickbloom.c qb_util.c -lm

$(BUILD)/fuzz_fasthash_var: tools/fuzz_fasthash_var.c quickbloom.c qb_util.c quickbloom.h | $(BUILD)
	$(CC) $(FUZZ_CFLAGS) -I. -o $@ tools/fuzz_fasthash_var.c quickbloom.c qb_util.c -lm

fuzz: $(BUILD)/fuzz_insert $(BUILD)/fuzz_fasthash_var

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
