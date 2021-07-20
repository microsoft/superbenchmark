---
id: release-process
---


# Release Process

We will follow below steps in a release cycle.

1. Develop new features in `feature` branch, then merge to `main` branch after pull request is approved and all test passed.
2. After code freeze, check out new `release/x.y` branch for bug bash.
3. Submit hotfix pull requests to `release/x.y` branch during bug fix stage. Pull requests for new features are still submitted to `main` branch.
4. After bug bash, tag the `release/x.y` branch with tag format `vx.y.z`. Write release note.
5. Generate corresponding docker image based on the latest release branch.
6. Cherry-pick and merge hotfix commits in `release` branch back to `main` branch.