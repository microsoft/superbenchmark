---
id: release-process
---


# Release Process

We will follow below steps in a release cycle.

1. Develop new features in `feature` branch, then merge to `main` branch after pull request is approved and all test passed.
2. After code freeze, pull `release` branch for bug bash.
3. Submit hotfix pull requests to `release` branch during bug bash. Pull requests for new features are still submitted to `main` branch.
4. Tag the `release` branch after bug bash. Write release note.
5. Cherry-pick and merge hotfix commits in `release` branch back to `main` branch.