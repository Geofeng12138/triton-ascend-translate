# Quick Start


## Project Introduction


**Triton-Ascend** is an optimized version of Triton adapted for Huawei Ascend processors, designed for efficient kernel auto-tuning, operator compilation, and deployment. By maintaining compatibility with Triton's core syntax and performing deep optimizations tailored to the characteristics of Ascend NPUs, it helps users quickly develop and deploy high-performance computing tasks on the Ascend platform.


This document uses the online installation and execution of a vector addition example via package deployment in the **Ubuntu 22.04** environment as a guide to help users quickly get started with **Triton-Ascend**. For more installation methods, please refer to the [Installation Guide](./installation_guide.md) document.


## Environment Setup


**Hardware Requirements**


Supported operating systems: Linux (aarch64/x86_64)


Supported Ascend products: Atlas A2/A3/950 series


Minimum hardware configuration: Single GPU with 32GB memory (recommended)


**Software Dependencies**


Determine the software versions of CANN, Python, and TorchNPU and install them. For reference, see the Ascend community official website "[CANN Quick Installation](https://www.hiascend.com/cann/download)" to complete the driver and firmware installation.


- CANN version: 9.0.0
- Python version: python3.11
- TorchNPU version: 2.7.1.post4


Note: For more compatibility information, please refer to the [Product Version Compatibility Table](./installation_guide.md#environment-preparation) in the installation guide.


## Quick Installation


```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```


## Quick Start


**Run the vector addition example in tutorials to verify the result**


Vector addition example: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
By comparing the output results of the Triton operator with PyTorch native computation, it is demonstrated that the Ascend NPU device can correctly invoke the Triton operator and ensure computational accuracy.


```bash
# 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 拉取triton-ascend源码仓及用例
git clone https://github.com/triton-lang/triton-ascend.git
# 运行tutorials实例
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```


Observing similar output indicates that the environment configuration is correct.


```shell
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

