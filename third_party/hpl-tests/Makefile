# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

OMPI_DIR=/opt/hpcx/ompi


all: ZEN3 ZEN4



ZEN3:
ifneq (,$(wildcard /opt/hpcx/ompi))
	cd ./hpl-2.3 && \
	make arch=Linux_zen3 omp=/opt/hpcx/ompi
else
	cd ./hpl-2.3 && \
	make arch=Linux_zen3 omp=/opt/ompi
endif 
	
ZEN4:
ifneq (,$(wildcard /opt/hpcx/ompi))
	cd ./hpl-2.3 && \
	make arch=Linux_zen4 omp=/opt/hpcx/ompi
else
	cd ./hpl-2.3 && \
	make arch=Linux_zen4 omp=/opt/ompi
endif 

clean:
	cd ./hpl-2.3 && \
	make clean arch=Linux_zen3 &&\
	make clean arch=Linux_zen4
