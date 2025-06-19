# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

GENFLAGS := -DSTREAM_ARRAY_SIZE=120000000
ZEN3FLAGS := -DSTREAM_ARRAY_SIZE=400000000 -march=znver3
ZEN4FLAGS := -DSTREAM_ARRAY_SIZE=800000000 -march=znver4
NEO2FLAGS := -DSTREAM_ARRAY_SIZE=240000000 -mcpu=neoverse-v2

GEN_OUTPUT := stream
ZEN3_OUTPUT := streamZen3
ZEN4_OUTPUT := streamZen4
NEO2_OUTPUT := streamNeo2

ALL_TARGETS := GEN

ARCH := $(shell uname -m)

# ARM64 build gets NEO2 added
ifeq ($(ARCH), aarch64)
  CC      := gcc
  CFLAGS  := -Ofast -fopenmp -mcmodel=large -fno-PIC -DNTIMES=200
  ALL_TARGETS += NEO2
else
  CC := /opt/AMD/aocc-compiler-4.0.0/bin/clang
  CFLAGS := -Ofast -mcmodel=large -mavx2 -ffp-contract=fast -lomp -fopenmp -fnt-store=aggressive -DNTIMES=10
  all: ZEN3 ZEN4 X86 NEO2
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

# clean up the generated files
clean:
	rm -f $(GEN_OUTPUT) $(ZEN3_OUTPUT) $(ZEN4_OUTPUT) $(NEO2_OUTPUT)
