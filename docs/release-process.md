---
id: release-process
---


# Release Process

We will follow below steps in a release cycle.

1. Develop new features in `feature` branch, then merge to `main` branch after pull request is approved and all test passed.
2. After code freeze, checkout new `release/X.Y` branch for bug bash.
3. Submit hotfix pull requests to `release/X.Y` branch during bug fix stage. Pull requests for new features are still submitted to `main` branch.
4. After bug bash, tag the `release/X.Y` branch with specific tag `vX.Y.Z`. (Where **X** is the major version, **Y** is the minor version, and **Z** is the patch version.)
5. Write release note and build Docker image for specific tag.
6. Cherry-pick and merge hotfix commits in `release/X.Y` branch back to `main` branch.