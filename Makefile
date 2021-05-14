# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CPPSOURCES := $(shell find $(CURDIR) -regextype posix-extended -regex '.*\.(c|cpp|h|hpp|cu)')

.PHONY: cpplint cppformat

cpplint:
	clang-format --verbose --dry-run --Werror $(CPPSOURCES)

cppformat:
	clang-format --verbose -i $(CPPSOURCES)
