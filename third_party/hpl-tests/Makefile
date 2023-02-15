# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

ZEN3_OUTPUT= xhpl_z3
ZEN4_OUTPUT= xhpl_z4 

all: ZEN3 ZEN4

ZEN3:
cd ./hpl-2.3 && \
make arch=Linux_zen3

	
ZEN4:
cd ./hpl-2.3 && \
make arch=Linux_zen4

clean:
cd ./hpl-2.3 && \
make clean arch=Linux_zen3 &&\
make clean arch=Linux_zen4
