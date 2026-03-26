---
id: run-superbench
---

# Run SuperBench

Having prepared benchmark configuration and inventory files,
you can start to run SuperBench over all managed nodes.

## Deploy

Leveraging `sb deploy` command, we can easily deploy SuperBench environment to all managed nodes.
After running the following command, SuperBench will automatically access all nodes, pull container image and prepare container.

```bash
sb deploy -f local.ini
```

Alternatively, to run on remote nodes, use the corresponding inventory file instead.

If you are using password for SSH and cannot specify private key in inventory,
or your private key requires a passphase before use, you can do
```bash
sb deploy -f remote.ini --host-password [password]
```

:::note Note
You should deploy corresponding Docker image to use release version, for example,

`sb deploy -f local.ini -i superbench/superbench:v0.12.0-cuda12.9`

You should note that version of git repo only determines version of sb CLI, and not the sb container. You should define the container version even if you specified a release version for the git clone.

:::

## Run

After deployment, you can start to run the SuperBench benchmarks on all managed nodes using `sb run` command.

```bash
sb run -f local.ini -c resnet.yaml
```

:::tip TIP
For environments that cannot start containers through `sb deploy`, e.g., a Kubernetes cluster.
You can create a privileged container with `superbench/superbench` image, skip `sb deploy`, and run `sb run` directly inside the container with `--no-docker` argument:
`sb run --no-docker -l localhost -c resnet.yaml`.

:::

## Using `--no-docker` on Remote Nodes

When running `sb run` with `--no-docker` on **remote nodes** (via `--host-file` or `--host-list`), the following requirements apply:

1. **SuperBench must be pre-installed on each remote node.** The `sb` CLI binary and its dependencies must be available in the PATH on every target host. Running without Docker means Ansible will SSH into each node and execute `sb exec` directly; if `sb` is not installed, you will see `command not found` (exit code 127).

2. **Deployment options:**
   - **Option A:** Extract the contents of the `superbench/superbench` Docker image onto each node (e.g., copy binaries, Python environment, and micro-benchmark executables to a consistent path), then ensure `sb` is in PATH.
   - **Option B:** Install SuperBench from source or pip on each node, and build/install the required micro-benchmark binaries (see `third_party/` and build instructions).
   - **Option C (requires Docker on remote nodes):** If Docker is available on the remote nodes for deployment but you still want to execute benchmarks without containers, you can first use `sb deploy` to pull the image and prepare the container, then manually extract the container filesystem to the host and run subsequent `sb run --no-docker` commands against that host installation.

3. **Environment configuration:** Ensure the `SB_MICRO_PATH` environment variable is set on each remote node so that it matches the on-host installation path of SuperBench micro-benchmark binaries when using `--no-docker`. Alternatively, you can set the config key `superbench.env.SB_MICRO_PATH` via `--config-override` so that SuperBench exports this environment variable for remote executions.

4. **Use case:** `--no-docker` is intended for environments where Docker-in-Docker or nested containers are not supported (e.g., certain Kubernetes setups, HPC clusters with restricted container runtimes). For standard deployments, prefer `sb deploy` + `sb run` without `--no-docker`.
