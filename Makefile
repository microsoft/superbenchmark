# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CPPSOURCES := $(shell find $(CURDIR) -regextype posix-extended -regex '.*\.(c|cpp|h|hpp|cc|cxx|cu)')

.PHONY: cpplint cppformat

cpplint:
	clang-format --verbose --dry-run --Werror $(CPPSOURCES)

cppformat:
	clang-format --verbose -i $(CPPSOURCES)

cppbuild:
	cd ./superbench/benchmarks/ &&  bash build.sh
