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

python3 -m pip install -c requirements/ansible.txt -e .[develop]
make postinstall
```

Use a clean Python environment without the bundled `ansible` distribution.
The Linux control environment needs GNU `timeout`, `openssh-client` and `rsync`.

### Ansible Dependencies

[requirements/ansible.txt](../../requirements/ansible.txt) pins the engine,
runner and Jinja2 for each supported Python family. Other transitive dependencies
remain managed by pip; this is not a complete environment lock.
[Makefile](../../Makefile) pins the three collections, using a legacy set for
Python 3.7-3.10 and a modern set for Python 3.11+. The legacy Ansible 2.10 engine
is end-of-life and remains a compatibility bridge, not a long-term support promise.

`make postinstall` installs collections with at most three attempts, a
300-second timeout per attempt, and 10- and 20-second waits. Exhaustion fails
the build. Pip retains its own retry handling.

The main GPU image stage installs these dependencies before copying application
source or building benchmarks; CUDA 13.4's separate STREAM stage is unchanged.
The images reuse normal Docker layer caching and retain `PIP_CONSTRAINT` for
later pip commands. A cold build still needs the network; the existing Build
Image workflow is unchanged.

To update versions, edit the requirements file and Makefile together. Run
`python3 -m pytest -q tests/runner/test_ansible.py`, the existing Ansible
integration tests, and the affected image builds before merging. Check both
legacy and modern runtimes; local runner tests alone do not qualify GPU images
or remote deployment.

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
