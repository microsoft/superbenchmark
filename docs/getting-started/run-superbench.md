---
id: run-superbench
---

# Run SuperBench

Having prepared benchmark configuration and inventory files,
it will be easy to run SuperBench over all managed nodes.

## Deploy

We need to deploy SuperBench environments to all managed nodes first,
it will access all nodes, pull container image and prepare container.

```bash
sb deploy -f local.ini
```

Alternatively, to run on remote nodes, use the corresponding inventory file instead.

If you are using password for SSH and cannot specify private key in inventory,
or your private key requires a passphase before use, you can do
```bash
sb deploy -f remote.ini --host-password [password]
```

## Run

After deployment, you can start to run the SuperBench benchmarks on all managed nodes.

```bash
sb run -f local.ini -c resnet.yaml
```
