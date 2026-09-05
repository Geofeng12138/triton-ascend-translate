# Release Policy


## Version Numbering


Triton-Ascend follows the [PEP 440](https://peps.python.org/pep-0440/) version specification, with version numbers aligned with upstream Triton: `vMAJOR.MINOR.PATCH[rcN][.postN]`


- **MAJOR.MINOR**: Corresponds one-to-one with the upstream Triton version. For example, Triton-Ascend `3.2` is based on Triton `3.2`.
- **PATCH**: The `PATCH` version of Triton-Ascend may be higher than the upstream Triton, used for bug fixes or improvements at the `MAJOR.MINOR` level. For example, both Triton-Ascend `3.2.0` and `3.2.1` are based on Triton `3.2.0`.
- **rcN**: Release candidate version, released as needed for early community testing and feedback.
- **postN**: Subsequent patch for a released version, released as needed to fix issues in a stable version.


## Branch Strategy


- The `main` branch is the latest development branch, tracking the latest upstream Triton version.
- Each release version creates a corresponding release development branch (e.g., `release/3.2.x`), which has the same commit ID as the community release.
- Feature development should be carried out in a forked repository and merged into the Triton-Ascend repository via a `PR`.


**`main` branch mapping:**


| Triton-Ascend | Triton commit hash                                           | Python    | CANN  | PyTorch | LLVM commit hash                                             | Patch                                                        |
| ------------- | ------------------------------------------------------------ | --------- | ----- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `main`        | [cfc0a9d](https://github.com/triton-lang/triton-ascend/commit/cfc0a9d) | `3.9~3.13` | `9.0.0` | `2.7.1`   | [fad3272](https://github.com/llvm/llvm-project/commit/fad3272) | [fad3272.patch](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/llvm_patch/fad3272.patch) |


## Maintenance Branches and Lifecycle


Maintenance branch status includes:


- **Active**: Continuously receives bug fixes, feature enhancements, and security patches; will continue to evolve features or release new versions  
- **Maintenance**: Only accepts critical bug fixes and security patches; no longer releases feature enhancements  
- **End of Life**: No longer accepts any fixes; branch maintenance has ceased


| Branch             | Status   | Triton Version | Triton-Ascend Release              | Maintenance End |
| ----------------- | -------- | -------------- | ----------------------------------- | -------------- |
| `main`            | `Active` | `3.5.0`        | /                                   | /              |
| `release/3.2.1`   | `Active` | `3.2.0`        | `3.2.1`                             | /              |
| `release/3.2.x`   | `Maintenance` | `3.2.0`    | `3.2.0rc2`, `3.2.0rc3`, `3.2.0rc4`, `3.2.0` | /              |


## Release Cycle


- **Stable version**: Released according to the project version cadence; not every upstream Triton version will have a corresponding stable release.
- **rc version**: Released in sync with the upstream Triton version cadence for early user testing.
- **post version**: Released as needed to fix issues in existing stable versions.


### Release Timeline


| Date       | Event                        |
| ---------- | ---------------------------- |
| 2026-05-06 | Released stable version `3.2.1` |
| 2026-01-21 | Released stable version `3.2.0` |
| 2025-11-14 | Released preview version `3.2.0rc4` |
| 2025-11-12 | Released preview version `3.2.0rc3` |
| 2025-05-26 | Released preview version `3.2.0rc2` |


## Version Compatibility Matrix


| Triton-Ascend | Triton | Python              | CANN  | PyTorch | LLVM commit hash | LLVM Patch |
| ------------- | ------ | ------------------- | ----- | ------- | ---------------- | --------- |
| `3.2.1`       | `3.2.0` | `3.9`(x86), `3.10-3.13` | `9.0.0` | `2.7.1`   | `b5cc222`        | -         |
| `3.2.0`       | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -         |
| `3.2.0rc4`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -         |
| `3.2.0rc3`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -         |
| `3.2.0rc2`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -         |

