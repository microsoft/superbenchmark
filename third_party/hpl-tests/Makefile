# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# OS_SKU is inherited from parent Makefile, but provide fallback if run independently
OS_SKU ?= $(shell sort -r /etc/*-release | gawk 'match($$0, /^ID=(.*)$$/, a) { print toupper(a[1]); exit }')

# Dynamically find AMD software paths
ifeq ($(OS_SKU), AZURELINUX)
BLIS_PATH := $(shell find /opt -path "*/amd-blis" 2>/dev/null | head -1)
BLIS_LIB := $(shell find /opt -name "libblis-mt.so" 2>/dev/null | head -1)
AOCC_PATH := $(shell find /opt -path "*/aocc-compiler*/bin/clang" 2>/dev/null | head -1)
endif

all: ZEN3 ZEN4 CONFIGURE

CONFIGURE:
	cp ./hpl-2.3/setup/Make.Linux_ATHLON_CBLAS ./hpl-2.3/Make.Linux_zen3
	sed -i 's/.*ARCH   .*=.*/ARCH        =$$(arch)/' ./hpl-2.3/Make.Linux_zen3
	sed -i 's/.*TOPdir .*=.*/TOPdir      = ..\/..\/../' ./hpl-2.3/Make.Linux_zen3
	sed -i 's/.*MPdir .*=.*/MPdir        = $$(omp)/' ./hpl-2.3/Make.Linux_zen3
	sed -i 's/.*MPinc .*=.*/MPinc        = -I$$(MPdir)\/include/' ./hpl-2.3/Make.Linux_zen3
	sed -i 's/.*MPlib .*=.*/MPlib        = $$(MPdir)\/lib\/libmpi.so/' ./hpl-2.3/Make.Linux_zen3
ifeq ($(OS_SKU), AZURELINUX)
	sed -i 's|.*LAdir .*=.*|LAdir        = $(BLIS_PATH)|' ./hpl-2.3/Make.Linux_zen3
	sed -i 's|.*LAlib .*=.*|LAlib        = $(BLIS_LIB)|' ./hpl-2.3/Make.Linux_zen3
	sed -i 's|.*CC .*=.*|CC              = $(AOCC_PATH)|' ./hpl-2.3/Make.Linux_zen3	
	sed -i 's|.*LINKER .*=.*|LINKER      = $(AOCC_PATH)|' ./hpl-2.3/Make.Linux_zen3
else
	sed -i 's/.*LAdir .*=.*/LAdir        = \/opt\/AMD\/amd-blis/' ./hpl-2.3/Make.Linux_zen3
	sed -i 's/.*LAlib .*=.*/LAlib        = $$(LAdir)\/lib\/LP64\/libblis-mt.so/' ./hpl-2.3/Make.Linux_zen3
	sed -i 's/.*CC .*=.*/CC              = \/opt\/AMD\/aocc-compiler-4.0.0\/bin\/clang/' ./hpl-2.3/Make.Linux_zen3	
	sed -i 's/.*LINKER .*=.*/LINKER      = \/opt\/AMD\/aocc-compiler-4.0.0\/bin\/clang/' ./hpl-2.3/Make.Linux_zen3
endif
	sed -i 's/LAinc .*=/LAinc              = -I$$(LAdir)\/lib\/include/' ./hpl-2.3/Make.Linux_zen3
	sed -i 's/.*CCFLAGS .*=.*/CCFLAGS      = $$(HPL_DEFS) -march=znver3 -fomit-frame-pointer -O3 -funroll-loops/' ./hpl-2.3/Make.Linux_zen3
	cp ./hpl-2.3/Make.Linux_zen3 ./hpl-2.3/Make.Linux_zen4
	sed -i 's/.*CCFLAGS .*=.*/CCFLAGS      = $$(HPL_DEFS) -march=znver4 -fomit-frame-pointer -O3 -funroll-loops/' ./hpl-2.3/Make.Linux_zen4
ZEN3: CONFIGURE
ifneq (,$(wildcard /opt/hpcx/ompi))
	cd ./hpl-2.3 && \
	make arch=Linux_zen3 omp=/opt/hpcx/ompi
else
	cd ./hpl-2.3 && \
	make arch=Linux_zen3 omp=/opt/ompi/
endif 
	
ZEN4: CONFIGURE
ifneq (,$(wildcard /opt/hpcx/ompi))
	cd ./hpl-2.3 && \
	make arch=Linux_zen4 omp=/opt/hpcx/ompi
else
	cd ./hpl-2.3 && \
	make arch=Linux_zen4 omp=/opt/ompi/
endif 

clean:
	cd ./hpl-2.3 && \
	make clean arch=Linux_zen3 &&\
	make clean arch=Linux_zen4
