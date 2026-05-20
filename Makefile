# quickbloom Makefile
#
# Targets:
#   make            -- build the static and shared libraries (default)
#   make lib        -- build build/libquickbloom.{a,so}
#   make test       -- build and run the native C test binary
#   make example    -- build the hello_quickbloom example
#   make bench      -- run the Python benchmark sweep
#   make install    -- install header + libraries to $(PREFIX) (default /usr/local)
#   make clean      -- remove build artefacts
#
# Variables (override on the command line):
#   CC=clang        -- C compiler (default: cc)
#   PREFIX=...      -- install root (default: /usr/local)
#   CFLAGS_EXTRA=.. -- additional compiler flags

CC      ?= cc
PREFIX  ?= /usr/local
BUILD   := build

CFLAGS_BASE := -O3 -mavx2 -mbmi2 -mfma -maes -fPIC -Wall -Wextra -std=c11
CFLAGS      := $(CFLAGS_BASE) $(CFLAGS_EXTRA)

# Each variant compiles bloom_sbbf.c or bloom_batched.c with a different
# QB_NS and PREFETCH_LOOKAHEAD. All three end up in the same library.
OBJS := \
	$(BUILD)/single_key.o \
	$(BUILD)/unified.o    \
	$(BUILD)/batched.o

LIB_SO := $(BUILD)/libquickbloom.so
LIB_A  := $(BUILD)/libquickbloom.a

.PHONY: all lib test example bench install clean

all: lib

lib: $(LIB_SO) $(LIB_A)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/single_key.o: bloom_sbbf.c bloom_single_key.c | $(BUILD)
	$(CC) $(CFLAGS) -DQB_NS=qb_single_key -DPREFETCH_LOOKAHEAD=0 -c bloom_sbbf.c -o $@

$(BUILD)/unified.o: bloom_sbbf.c bloom_unified.c | $(BUILD)
	$(CC) $(CFLAGS) -DQB_NS=qb_unified -DPREFETCH_LOOKAHEAD=8 -c bloom_sbbf.c -o $@

$(BUILD)/batched.o: bloom_batched.c | $(BUILD)
	$(CC) $(CFLAGS) -c bloom_batched.c -o $@

$(LIB_SO): $(OBJS)
	$(CC) -shared -o $@ $(OBJS)

$(LIB_A): $(OBJS)
	ar rcs $@ $(OBJS)

# Native C tests. Links against the static library so a system without
# the shared lib installed can still run them.
$(BUILD)/test_quickbloom: test/test_quickbloom.c $(LIB_A) quickbloom.h | $(BUILD)
	$(CC) $(CFLAGS) -I. -o $@ test/test_quickbloom.c $(LIB_A)

test: $(BUILD)/test_quickbloom
	$<

# Example. Same linkage shape as a consumer would use.
$(BUILD)/hello_quickbloom: examples/hello_quickbloom.c $(LIB_A) quickbloom.h | $(BUILD)
	$(CC) $(CFLAGS) -I. -o $@ examples/hello_quickbloom.c $(LIB_A)

example: $(BUILD)/hello_quickbloom
	$<

# Python benchmark sweep. Compiles each variant's .c separately the way
# it always has — independent of the library build above.
bench:
	python3 bench_all.py

install: lib
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib
	install -m 644 quickbloom.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(LIB_A)     $(DESTDIR)$(PREFIX)/lib/
	install -m 755 $(LIB_SO)    $(DESTDIR)$(PREFIX)/lib/

clean:
	rm -rf $(BUILD)
