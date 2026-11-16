---
id: development
---

# Development

If you want to develop new feature, please follow below steps to set up development environment.

We suggest you to use [Visual Studio Code](https://vscode.github.com/) and install the recommended extensions for this project.
You can also develop online with [GitHub Codespaces](https://github.com/codespaces).

## Check Environment

Follow [System Requirements](../getting-started/installation.mdx).

## Set Up

```bash
git clone --recurse-submodules -j8 https://github.com/microsoft/superbenchmark
cd superbenchmark

python3 scripts/install_ansible.py
python3 -m pip install -c requirements/ansible.txt -e .[develop]
python3 scripts/install_ansible.py --check
```

The POSIX control environment also needs `openssh-client` and `rsync`.
Use a clean Python environment; do not install the bundled `ansible` distribution
alongside `ansible-base` or `ansible-core`.

### Ansible Dependency Profiles

The Python freeze in `requirements/ansible.txt` includes the engine, runner and
their transitive Python dependencies. Collections are pinned separately:

| Control Python | Engine | Runner | Collections |
| --- | --- | --- | --- |
| 3.7-3.10 | ansible-base 2.10.17 | 2.3.1 | `ansible-legacy.yml` |
| 3.11 | ansible-core 2.19.15 | 2.4.3 | `ansible-modern.yml` |
| 3.12 | ansible-core 2.21.6 | 2.4.3 | `ansible-modern.yml` |
| 3.13+ | ansible-core 2.22.0 | 2.4.3 | `ansible-modern.yml` |

These are deployment inputs, not a promise that every future Python release is
supported. The old 2.10 engine is end-of-life; retaining it is a compatibility
bridge for existing images, not a long-term support solution. Engine and Python
support lifetimes must be reviewed when updating profiles. Control Python and
managed-host Python have separate compatibility requirements.

`make postinstall` installs and checks the selected profile. The installer uses
the current Python interpreter, checks collection engine requirements, and
reports the paths of the plugins actually loaded. `--check` is read-only and
does not contact Galaxy. Use it after installing SuperBench to detect dependency
drift or a different collection shadowing the pinned copy.

Only the Galaxy acquisition command has an external retry loop: at most three
attempts, each limited to 300 seconds, with 10- and 20-second waits. Logs are
preserved and exhaustion fails the build. Pip retains its own network handling;
do not add another retry around the entire Docker build or workflow.

To update dependencies, edit `requirements/ansible.in`, regenerate the Python
freeze, update the collection requirements, and validate every affected profile:

```bash
uv pip compile --universal --python-version 3.7 requirements/ansible.in --output-file requirements/ansible.txt
python3 scripts/install_ansible.py
python3 -m pytest -q tests/ansible/test_dependencies.py tests/runner/test_ansible.py
```

The local collection test generates disposable keys without logging private key
contents, checks rank lookup and idempotency, and exercises the production
result-fetch playbook on local files. It does not replace remote deployment,
container-internal `--no-docker`, or GPU image validation.
Version pins and cache reuse do not make a cold build offline. Hash-verified
offline artifacts and broader image reproducibility are outside this change.

### Dependency Layers in GPU Images

Active GPU Dockerfiles install the profile in the `ansible-dependencies` stage,
before application source and benchmark compilation. Both CUDA 13.4 build
branches inherit that stage so its validation precedes the separate STREAM
compiler build as well. Later pip operations inherit `PIP_CONSTRAINT`, and
application installation is followed by a read-only profile check and local
collection tests rather than another online collection installation.

The Build Image workflow checks the actual stage on dependency-related PRs:
a cold build must execute installation, an application-only edit must reuse it,
and a requirements edit must invalidate it. Other PRs do not force cold
dependency downloads. The check temporarily edits and restores two input files;
run it only in a disposable checkout:

```bash
bash tests/ansible/check_build_cache.sh dockerfile/cuda11.1.1.dockerfile linux/amd64
```

Changing the parent image or dependency inputs can still invalidate the layer.
The usual complete image matrix remains necessary to qualify both architectures
and detect conflicts with preinstalled GPU-framework dependencies.

## Lint and Test

Format code using yapf.
```bash
python3 setup.py format
```

Check code style with mypy and flake8
```bash
python3 setup.py lint
```

Run unit tests.
```bash
python3 setup.py test
```

## Submit a Pull Request

Please install `pre-commit` before `git commit` to run all pre-checks.

```bash
pre-commit install --install-hooks
```

Open a pull request to main branch on GitHub.
