# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# OS_SKU is inherited from parent Makefile, but provide fallback if run independently
OS_SKU ?= $(shell sort -r /etc/*-release | gawk 'match($$0, /^ID=(.*)$$/, a) { print toupper(a[1]); exit }')

GENFLAGS := -DSTREAM_ARRAY_SIZE=120000000
ZEN3FLAGS := -DSTREAM_ARRAY_SIZE=400000000 -march=znver3
ZEN4FLAGS := -DSTREAM_ARRAY_SIZE=800000000 -march=znver4
NEO2FLAGS := -DSTREAM_ARRAY_SIZE=120000000 -mcpu=neoverse-v2

GEN_OUTPUT := stream
ZEN3_OUTPUT := streamZen3
ZEN4_OUTPUT := streamZen4
NEO2_OUTPUT := streamNeo2

ALL_TARGETS := GEN

ARCH := $(shell uname -m)

# ARM64 build gets NEO2 added
ifeq ($(ARCH), aarch64)
  CC      := gcc
  CFLAGS  := -Ofast -fopenmp -DNTIMES=200
  ALL_TARGETS += NEO2
endif

# AMD AOCC clang present? add ZEN3 and ZEN4
# Find AOCC compiler installation
ifeq ($(OS_SKU), AZURELINUX)
AOCC_CLANG := $(shell find /opt -path "*/aocc-compiler*/bin/clang" 2>/dev/null | head -1)
else
AOCC_CLANG := $(wildcard /opt/AMD/aocc-compiler-4.0.0/bin/clang)
endif

# Configure compiler if AOCC is found
ifneq ($(AOCC_CLANG),)
  CC      := $(AOCC_CLANG)
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
