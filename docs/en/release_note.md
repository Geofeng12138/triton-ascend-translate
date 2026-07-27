# Triton-Ascend Release Notes


The Triton-Ascend version provides a stable snapshot of the codebase, packaged as binary distributions that can be easily installed via PyPI. Additionally, a version release signifies that the development team can formally announce to the community the availability of new features, completed improvements, and changes that may affect users (such as breaking changes).


## Release Compatibility Matrix


The following is the release compatibility matrix for the Triton-Ascend version:


| Triton-Ascend Version | Python Version | Manylinux Version | Hardware Platform | Hardware Product |
| --- | --- | --- | --- | --- |
| 3.2.0 | >=3.9, <=3.11 | glibc 2.27+, x86-64, aarch64 | Ascend NPU | Atlas A2/A3 |


## Release Plan


Below is the release plan for Triton-Ascend. Please note: patch versions are optional.


| Major Version | Release Branch Cut Date | Release Date | Patch Release Date |
| --- | --- | --- | --- |
| 3.2.0 | December 08, 2025 | January 2026 | --- |


## Version Highlights


### Triton-Ascend 3.2.0


**First Release: Ascend NPU Support**


Triton-Ascend 3.2.0 is the first Triton version to officially support Huawei Ascend NPU. This version is based on the Triton 3.2.0 community release and is specifically adapted to the Ascend NPU hardware architecture.


#### Key Features


1. **Ascend NPU Full-Stack Support**
   - Complete compilation pipeline from Triton IR to NPU instruction set
   - Support for all Triton Ops


2. **Performance Optimization**
   - NPU-specific kernel optimization
   - CV computation optimization


3. **Developer Tools**
   - Supports comprehensive debug output
   - Dump of intermediate compilation artifacts


#### Known Limitations


1. **Data Types**: Support for some data types is still being improved.
2. **Operator Coverage**: The set of supported operators is continuously being expanded.


#### Migration Guide


For existing Triton GPU users migrating to Ascend NPU, see [GPU Triton Operator Migration](./migration_guide/migrate_from_gpu.md) for details.

