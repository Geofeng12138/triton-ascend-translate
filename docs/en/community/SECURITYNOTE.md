# Security Notice


## System Security Hardening


It is recommended that users enable ASLR (**Address Space Layout Randomization**) in the system, with level 2 (full randomization mode) preferred. Configuration can be performed as follows:


echo 2 > /proc/sys/kernel/randomize_va_space


## Recommended Runtime User


For security and least privilege considerations, it is not recommended to use Triton-Ascend with administrator accounts such as root.


## File Permission Control


1. It is recommended that users implement security measures such as access control for sensitive files like personal private data and commercial assets. The permissions should be set with reference to the [File Permission Reference](#file-permission-reference).


2. Users need to implement proper permission control during installation and usage. It is recommended to refer to the [File Permission Reference](#文件权限参考) for configuration.


### File Permission Reference


| Type | Linux Permission Reference Maximum |
|------|-----------------------------------|
| User home directory | 750 (rwxr-x---) |
| Program files (including scripts, libraries, etc.) | 550 (r-xr-x---) |
| Program file directory | 550 (r-xr-x---) |
| Configuration files | 640 (rw-r-----) |
| Configuration file directory | 750 (rwxr-x---) |
| Log files (completed or archived) | 440 (r--r-----) |
| Log files (currently being written) | 640 (rw-r-----) |
| Log file directory | 750 (rwxr-x---) |
| Debug files | 640 (rw-r-----) |
| Debug file directory | 750 (rwxr-x---) |
| Temporary file directory | 750 (rwxr-x---) |
| Maintenance and upgrade file directory | 770 (rwxrwx---) |
| Business data files | 640 (rw-r-----) |
| Business data file directory | 750 (rwxr-x---) |
| Key components, private keys, certificates, ciphertext file directory | 700 (rwx------) |
| Key components, private keys, certificates, encrypted ciphertext | 600 (rw-------) |
| Encryption/decryption interfaces, encryption/decryption scripts | 500 (r-x------) |


## Build Security Notice


Triton-Ascend supports source code compilation and installation. During compilation, it downloads dependent third-party libraries and executes build shell scripts, generating temporary program files and build directories. Users can manage permissions on files within the source code directory as needed to reduce security risks.


## Public Network Address Declaration


In the configuration files and scripts of Triton-Ascend, there are [public network addresses](#public-network-addresses).


### Public Network Address


| Type | Open Source Code URL | File Name | Public IP Address/Public URL/Domain/Email Address | Purpose Description |
|------|---------------------|-----------|---------------------------------------------------|-------------------|
| Open Source Import | <https://github.com/triton-lang/triton.git> | .gitmodules | <https://github.com/triton-lang/triton.git> | Triton source code repository URL |
| Open Source Import | <https://gitcode.com/Ascend/AscendNPU-IR.git> | .gitmodules | <https://gitcode.com/Ascend/AscendNPU-IR.git> | AscendNPU IR source code repository URL |
| Self-developed | Not applicable | docker/devdocker/setup_triton-ascend_dev.sh | <https://gitcode.com/Ascend/triton-ascend.git> | Triton-Ascend source code repository URL |
| Self-developed | Not applicable | ascend/examples/generalization_cases/run_daily.sh & scripts/prepare_build.sh | <https://gitee.com/shijingchang/triton.git> | Build dependency code repository |
| Self-developed | Not applicable | setup.py | <https://gitcode.com/Ascend/triton-ascend/> | Triton-Ascend source code repository URL |
| Open Source Import | <https://gitclone.com> | scripts/prepare_build.sh | <https://gitclone.com/github.com/llvm/llvm-project.git> | Dependent LLVM source code repository |
| Open Source Import | <https://repo.huaweicloud.com> | scripts/prepare_build.sh | <https://repo.huaweicloud.com/repository/pypi/simple> | Used to configure pybind11 download link |
| Open Source Import | <https://pypi.tuna.tsinghua.edu.cn> | docker/devdocker/triton-ascend_dev.dockerfile | <https://pypi.tuna.tsinghua.edu.cn/simple> | Python pip source configuration |
| Open Source Import | <https://triton-ascend-artifacts.obs.myhuaweicloud.com> | setup.py | `https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz` | Used to download precompiled LLVM tools |
| Open Source Import | <https://bootstrap.pypa.io/get-pip.py> | docker/develop_env.dockerfile | <https://bootstrap.pypa.io/get-pip.py> | Used for automated pip installation |
| Open Source Import | <https://llvm.org/LICENSE.txt> | third_party/ascend/include/Dialect/TritonAscend/IR/ & third_party/ascend/lib/Dialect/TritonAscend/IR/ | <https://llvm.org/LICENSE.txt> | Apache License link |
| Open Source Import | <https://netlib.org/cephes/> | third_party/ascend/language/cann/libdevice.py | <https://netlib.org/cephes/> | Function source declaration |

