# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

GENFLAGS := -DSTREAM_ARRAY_SIZE=120000000
ZEN3FLAGS := -DSTREAM_ARRAY_SIZE=400000000 -march=znver3
ZEN4FLAGS := -DSTREAM_ARRAY_SIZE=800000000 -march=znver4
NEO2FLAGS := -DSTREAM_ARRAY_SIZE=120000000 -mcpu=neoverse-v2
# Olympus has 328 MiB of L3 (352 MiB of L2) across 2 sockets; 400M doubles (~3 GiB per array) is
# more than 4x the combined cache. The arrays exceed 4 GiB in total, so use the large code model
# (which GCC does not support together with PIE).
OLYMPUSFLAGS := -DSTREAM_ARRAY_SIZE=400000000 -mcpu=olympus -mcmodel=large -fno-pie -no-pie
NATIVEFLAGS := -DSTREAM_ARRAY_SIZE=120000000 -mcpu=native

GEN_OUTPUT := stream
ZEN3_OUTPUT := streamZen3
ZEN4_OUTPUT := streamZen4
NEO2_OUTPUT := streamNeo2
OLYMPUS_OUTPUT := streamOlympus
NATIVE_OUTPUT := streamNative

ALL_TARGETS := GEN

ARCH := $(shell uname -m)

# ARM64 build gets NEO2 and NATIVE, plus OLYMPUS when supported by the compiler
ifeq ($(ARCH), aarch64)
  CC      := gcc
  CFLAGS  := -Ofast -fopenmp -DNTIMES=200
  ALL_TARGETS += NEO2 NATIVE
  OLYMPUS_SUPPORTED := $(shell printf 'int main(void) { return 0; }\n' | \
    $(CC) $(OLYMPUSFLAGS) -x c -c -o /dev/null - >/dev/null 2>&1 && echo yes)
  ifeq ($(OLYMPUS_SUPPORTED),yes)
    ALL_TARGETS += OLYMPUS
  endif
endif

# AMD AOCC clang present? add ZEN3 and ZEN4
ifneq ("$(wildcard /opt/AMD/aocc-compiler-4.0.0/bin/clang)","")
  CC      := /opt/AMD/aocc-compiler-4.0.0/bin/clang
  CFLAGS  := -Ofast -mcmodel=large -mavx2 -ffp-contract=fast -lomp -fopenmp \
             -fnt-store=aggressive -DNTIMES=10
  ALL_TARGETS += ZEN3 ZEN4
endif

# the one all: definition
all: $(ALL_TARGETS)

GEN:
	$(CC) $(CFLAGS) $(GENFLAGS) stream.c -o $(GEN_OUTPUT)
ZEN3:
	$(CC) $(CFLAGS) $(ZEN3FLAGS) stream.c -o $(ZEN3_OUTPUT)
ZEN4:
	$(CC) $(CFLAGS) $(ZEN4FLAGS) stream.c -o $(ZEN4_OUTPUT)
NEO2:
	$(CC) $(CFLAGS) $(NEO2FLAGS) stream.c -o $(NEO2_OUTPUT)
OLYMPUS:
	$(CC) $(CFLAGS) $(OLYMPUSFLAGS) stream.c -o $(OLYMPUS_OUTPUT)
NATIVE:
	$(CC) $(CFLAGS) $(NATIVEFLAGS) stream.c -o $(NATIVE_OUTPUT)

# clean up the generated files
clean:
	rm -f $(GEN_OUTPUT) $(ZEN3_OUTPUT) $(ZEN4_OUTPUT) $(NEO2_OUTPUT) $(OLYMPUS_OUTPUT) $(NATIVE_OUTPUT)
