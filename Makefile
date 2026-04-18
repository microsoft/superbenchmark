# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CPPSOURCES := $(shell find $(CURDIR) \
	-path $(CURDIR)/.git -prune -o \
	-path $(CURDIR)/.venv -prune -o \
	-path $(CURDIR)/build -prune -o \
	-path $(CURDIR)/third_party -prune -o \
	-regextype posix-extended -regex '.*\.(c|cpp|h|hpp|cc|cxx|cu)' -print)
CLANG_FORMAT ?= clang-format-14

.PHONY: cpplint cppformat cppbuild thirdparty postinstall

cpplint:
	$(CLANG_FORMAT) --verbose --dry-run --Werror $(CPPSOURCES)

cppformat:
	$(CLANG_FORMAT) --verbose -i $(CPPSOURCES)

cppbuild:
	cd ./superbench/benchmarks/ && bash build.sh

directxbuild:
	cd ./superbench/benchmarks/ && build.bat

thirdparty:
	cd ./third_party/ && make all

postinstall:
ifeq ($(shell which ansible-galaxy),)
	$(error 'Cannot find ansible-galaxy')
else
	ansible-galaxy collection install ansible.posix ansible.utils community.crypto
endif
