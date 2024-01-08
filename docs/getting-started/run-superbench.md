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

`sb deploy -f local.ini -i superbench/superbench:v0.10.0-cuda12.2`

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
