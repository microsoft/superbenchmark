#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -euo pipefail

usage() {
    echo "Usage: $0 DOCKERFILE PLATFORM"
    echo "Check the actual Ansible stage's cold build and cache invalidation in a disposable CI checkout."
}

if [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

cd "$(dirname "$0")/../.."
dockerfile="$1"
platform="$2"
if [[ ! -f "$dockerfile" || ! -f requirements/ansible.txt || ! -f README.md ]]; then
    echo "Run this check against a SuperBench Dockerfile in a complete checkout." >&2
    exit 2
fi

scratch="$(mktemp -d)"
cp README.md "$scratch/README.md"
cp requirements/ansible.txt "$scratch/ansible.txt"
cleanup() {
    cp "$scratch/README.md" README.md
    cp "$scratch/ansible.txt" requirements/ansible.txt
    rm -f "$scratch/README.md" "$scratch/ansible.txt" "$scratch/build.log"
    rmdir "$scratch"
}
trap cleanup EXIT

check_build() {
    expected="$1"
    shift
    if ! docker buildx build --file "$dockerfile" --platform "$platform" \
        --target ansible-dependencies --progress=plain "$@" . > "$scratch/build.log" 2>&1; then
        cat "$scratch/build.log" >&2
        return 1
    fi
    cat "$scratch/build.log"
    step="$(awk '/] RUN python3 \/opt\/sb-ansible\/scripts\/install_ansible.py$/ {print $1; exit}' "$scratch/build.log")"
    if [[ -z "$step" ]]; then
        echo "The build did not report the Ansible installation step." >&2
        return 1
    fi
    if [[ "$expected" == "cached" ]]; then
        if ! grep -Fxq "$step CACHED" "$scratch/build.log"; then
            echo "An application-only change invalidated the Ansible dependency layer." >&2
            return 1
        fi
    elif grep -Fxq "$step CACHED" "$scratch/build.log" || ! grep -Fq "$step DONE" "$scratch/build.log"; then
        echo "The Ansible installation did not execute for a cold or changed-dependency build." >&2
        return 1
    fi
}

check_build rebuilt --no-cache
printf '\n<!-- Ansible dependency cache probe -->\n' >> README.md
check_build cached
printf '\n# Ansible dependency cache probe\n' >> requirements/ansible.txt
check_build rebuilt
echo "Verified cold installation, application-only cache reuse, and dependency-change invalidation."
