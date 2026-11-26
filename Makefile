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
	@command -v ansible-galaxy >/dev/null || { echo "Cannot find ansible-galaxy" >&2; exit 1; }
	@profile=$$(python3 -c 'import sys; print("legacy" if sys.version_info < (3, 11) else "modern")') || exit $$?; \
	if [ "$$profile" = legacy ]; then \
		collections="ansible.posix:1.5.4 ansible.utils:2.12.0 community.crypto:2.16.0"; \
	else \
		collections="ansible.posix:2.2.2 ansible.utils:6.1.1 community.crypto:3.5.1"; \
	fi; \
	for attempt in 1 2 3; do \
		timeout 300 ansible-galaxy collection install $$collections && exit 0; \
		status=$$?; \
		echo "Ansible collections: attempt $$attempt/3 failed (exit $$status)." >&2; \
		if [ $$attempt -eq 3 ]; then exit $$status; fi; \
		echo "Retrying in $$((attempt * 10)) seconds." >&2; \
		sleep $$((attempt * 10)) || exit $$?; \
	done
