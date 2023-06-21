# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CPPSOURCES := $(shell find $(CURDIR) -regextype posix-extended -regex '.*\.(c|cpp|h|hpp|cc|cxx|cu)')

.PHONY: cpplint cppformat cppbuild thirdparty postinstall

cpplint:
	clang-format --verbose --dry-run --Werror $(CPPSOURCES)

cppformat:
	clang-format --verbose -i $(CPPSOURCES)

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
