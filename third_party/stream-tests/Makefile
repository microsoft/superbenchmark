# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

GENFLAGS := -DSTREAM_ARRAY_SIZE=400000000
ZEN3FLAGS := -DSTREAM_ARRAY_SIZE=400000000 -march=znver3
ZEN4FLAGS := -DSTREAM_ARRAY_SIZE=800000000 -march=znver4
NEO2FLAGS := -DSTREAM_ARRAY_SIZE=120000000 -mcpu=neoverse-v2

GEN_OUTPUT := streamx86.exe
ZEN3_OUTPUT := streamZen3.exe
ZEN4_OUTPUT := streamZen4.exe
NEO2_OUTPUT := streamNeo2.exe

ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
CFLAGS := -Ofast -fopenmp -DNTIMES=200
CC := gcc
all: NEO2
else
CC := /opt/AMD/aocc-compiler-4.0.0/bin/clang
CFLAGS := -Ofast -mcmodel=large -mavx2 -ffp-contract=fast -lomp -fopenmp -fnt-store=aggressive -DNTIMES=10
all: ZEN3 ZEN4 X86
endif

ZEN3: stream.c
	$(CC) $(CFLAGS) $(ZEN3FLAGS) stream.c -o $(ZEN3_OUTPUT)
ZEN4:
	$(CC) $(CFLAGS) $(ZEN4FLAGS) stream.c -o $(ZEN4_OUTPUT)
X86:
	$(CC) $(CFLAGS) $(GENFLAGS) stream.c -o $(GEN_OUTPUT)
NEO2:
	$(CC) $(CFLAGS) $(NEO2FLAGS) stream.c -o $(NEO2_OUTPUT)

ifeq ($(ARCH), aarch64)
clean:
	rm $(NEO2_OUTPUT)
else
clean:
	rm $(GEN_OUTPUT) $(ZEN3_OUTPUT) $(ZEN4_OUTPUT)
endif
