# Triton-Ascend max_autotune Usage Guide


## Document Positioning


This document is intended for users who are already familiar with the automatic tuning mechanism of Triton-Ascend, and introduces advanced usage of the `max_autotune` decorator.


- The differences and connections between `max_autotune` and the standard `@triton.autotune`;
- How to use the Cartesian product expansion mechanism to batch generate candidate configurations;
- The tuning parameters supported by different `kernel_type` and their applicable scenarios;
- When to choose `max_autotune` over manually writing configuration lists.


## Quick Start


`max_autotune` is an extended version of the `@triton.autotune` decorator. It allows each base configuration to be expanded via a Cartesian product with additional tuning parameters before autotuning, thereby significantly reducing the number of configurations that users need to manually enumerate.


### 1. Basic Usage


```python
import triton
import triton.language as tl
from triton.backends.ascend.runtime import max_autotune


@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128}),
        triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 256}),
    ],
    key=["M", "N"],
    kernel_type="mix",
    # 额外的调优参数，每个值必须是列表
    enable_hivm_auto_cv_balance=[True, False],
    tile_mix_vector_loop=[2, 4],
)
@triton.jit
def kernel(
    a_ptr,
    b_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    **META
):
    ...
```


The number of equivalent configurations after expanding the above configuration is:


- Basic configuration: 2
- User-provided tuning parameters: `enable_hivm_auto_cv_balance` (2 types), `tile_mix_vector_loop` (2 types)
- Parameters using default values: `set_workspace_multibuffer` (2 types), `tile_mix_cube_loop` (2 types), 1 type for each of the other parameters


Total number of configurations: `2 × 2 × 2 × 2 × 2 = 32` configurations.


Note: `kernel_type="mix"` supports many parameters. Parameters not explicitly provided will use default values during expansion. If you want a specific parameter to not participate in expansion, you can fix its value in the `kwargs` of the base `Config`.


### 2. Relationship between `max_autotune` and `@triton.autotune`


The `max_autotune` essentially first calls `get_max_configs` to expand the base configuration, and then passes the expanded configuration list to the standard `@triton.autotune`. Therefore:


- All parameters supported by `@triton.autotune` (such as `key`, `prune_configs_by`, `reset_to_zero`, etc.) are also valid in `max_autotune`;
- `max_autotune` additionally introduces the `kernel_type` parameter and the `**tuning_params` tuning parameter space;
- Ultimately, `@triton.autotune` still handles benchmarking, optimization, and caching.


### 3. Must import the Ascend backend extension


Just like using `configs=[]` for automatic tiling, `max_autotune` requires support from the Ascend backend extension. `max_autotune` needs to be imported separately from the Ascend backend module:


```python
from triton.backends.ascend.runtime import max_autotune
```


## Kernel Types and Supported Parameters


`max_autotune` distinguishes different types of operators through the `kernel_type` parameter, with each type supporting a different set of tuning parameters.


### Parameter Support Matrix


| Parameter | cube | mix | vector | Default | Valid Values | Description |
|-----------|:----:|:---:|:------:|--------|--------------|-------------|
| `num_stages` | ✅ | ✅ | ✅ | `[2]` | `[1, 2]` | Number of pipeline stages |
| `unit_flag` | ✅ | ✅ | ❌ | `[False]` | Boolean list | Cube-related synchronization optimization for data movement |
| `limit_auto_multi_buffer_of_local_buffer` | ✅ | ✅ | ❌ | `["no-l0c"]` | `["no-limit", "no-l0c"]` | Configure the scope of automatic multi-buffer for local buffer |
| `limit_auto_multi_buffer_only_for_local_buffer` | ❌ | ✅ | ❌ | `[False]` | Boolean list | Restrict automatic multi-buffer to only apply to local buffer |
| `set_workspace_multibuffer` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4]` | Configure workspace multi-buffer level |
| `enable_hivm_auto_cv_balance` | ❌ | ✅ | ❌ | `[True]` | Boolean list | Enable or disable automatic CV balance |
| `tile_mix_vector_loop` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4, 8]` | Configure the number of splits for Vector loop |
| `tile_mix_cube_loop` | ❌ | ✅ | ❌ | `[2, 4]` | `[2, 4, 8]` | Configure the number of splits for Cube loop |
| `enable_ubuf_saving` | ❌ | ✅ | ✅ | `[True]` | Boolean list | Whether to enable ubuf saving |


### Kernel Type Description


- **cube**: Pure cube (matrix multiplication) operator, supporting the fewest tuning parameters;
- **vector**: Pure vector operator, only supports `num_stages` and `enable_ubuf_saving`;
- **mix**: Mixed cube+vector operator (default type), supporting the most complete set of tuning parameters.


## Parameter Value Priority and Expansion Logic


### Parameter Value Priority


The values of tuning parameters are determined according to the following priority:


1. **tuning_params parameter** (highest priority): list of candidate values passed via `**tuning_params`;
2. **Value in base configuration**: if the parameter already exists in the `kwargs` of the base `Config`, it is fixed to that value (converted to a single-element list);
3. **Default value** (lowest priority): obtained from the internal default value table.


### Expanded Example


Assume the following configuration:


```python
@max_autotune(
    configs=[triton.Config(kwargs={'BLOCK_M': 128}, num_stages=2)],
    key=["M", "N"],
    kernel_type="vector",
    num_stages=[1, 2],
)
@triton.jit
def kernel(...):
    ...
```


Expansion process:


1. The parameters supported by `kernel_type="vector"` are `num_stages` and `enable_ubuf_saving`;  
2. `num_stages` provides `[1, 2]` in `tuning_params`, with the highest priority;  
3. `enable_ubuf_saving` is not provided, so the default value `[True]` is used;  
4. After Cartesian product expansion, 2 configurations are obtained.


Expanding the result is equivalent to:


```python
configs=[
    triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': True}, num_stages=1),
    triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': True}, num_stages=2),
]
```


### Fixing Parameters in the Base Configuration


If you want a certain tuning parameter to not participate in the expansion, you can fix it directly in the base configuration:


```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'enable_ubuf_saving': False}),
    ],
    key=["M", "N"],
    kernel_type="vector",
    num_stages=[1, 2],
    # enable_ubuf_saving 已在基础配置中固定为 False，不会使用默认值 [True]
)
@triton.jit
def kernel(...):
    ...
```


## Usage Notes


### 1. Unsupported parameters will be ignored


If a parameter not supported by the current `kernel_type` is passed via `tuning_params`, a warning will be generated and the parameter will be ignored.


```python
# 警告：tile_mix_vector_loop 不支持 kernel_type="vector"
@max_autotune(
    configs=[...],
    key=["M"],
    kernel_type="vector",
    tile_mix_vector_loop=[2, 4],  # 将被忽略并产生警告
)
@triton.jit
def kernel(...):
    ...
```


### 2. Parameter values must be lists


Each value in `tuning_params` must be a list or tuple and cannot be empty.


```python
# 正确写法
enable_hivm_auto_cv_balance=[True, False]

# 错误写法：不是列表
enable_hivm_auto_cv_balance=True  # 将导致验证错误
```


### 3. Configuration Quantity Growth


The number of expanded configurations equals:  
Base configuration count × Π(length of each tuning_param list)


For example, 2 base configurations × 3 parameters (with list lengths of 2, 3, and 2 respectively) = 12 expanded configurations.


Too many configurations will increase the initial tuning time. It is recommended to reasonably control the parameter space based on actual needs.


### 4. Difference from `configs=[]`


`max_autotune` and `@triton.autotune(configs=[], ...)` are two different auto-tuning strategies:


| Feature | `max_autotune` | `@triton.autotune(configs=[])` |
|---------|----------------|--------------------------------|
| Tiling parameter generation | Requires user to specify in the base configuration | Automatically generated by Ascend backend |
| Compilation parameter tuning | Supported via `tuning_params` expansion | Passed through the `hints` parameter |
| Applicable scenario | When the Tiling parameter space is clearly known and compilation parameters need tuning | When Tiling parameters are also expected to be automatically generated |


## Advanced Usage


### 1. Combining Multiple Tuning Parameters


For mixed-type operators (`kernel_type="mix"`), multiple parameters can be tuned simultaneously:


```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 64, 'BLOCK_N': 64}),
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128}),
    ],
    key=["M", "N", "K"],
    kernel_type="mix",
    num_stages=[1, 2],
    enable_hivm_auto_cv_balance=[True, False],
    tile_mix_vector_loop=[2, 4, 8],
    tile_mix_cube_loop=[2, 4],
)
@triton.jit
def mixed_kernel(...):
    ...
```


Expanded configuration quantity calculation:


- Basic configuration: 2
- User-provided parameters: `num_stages` (2), `enable_hivm_auto_cv_balance` (2), `tile_mix_vector_loop` (3), `tile_mix_cube_loop` (2)
- Parameters using default values: `set_workspace_multibuffer` (default `[2, 4]` → 2), other parameters each 1


Total configuration count: `2 × 2 × 2 × 2 × 3 × 2 = 96` configurations.


### 2. Tuning for cube-type operators


Cube-type operators (such as pure matrix multiplication) support fewer parameters:


```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    ],
    key=["M", "N", "K"],
    kernel_type="cube",
    num_stages=[1, 2],
    unit_flag=[True, False],
    limit_auto_multi_buffer_of_local_buffer=["no-limit", "no-l0c"],
)
@triton.jit
def matmul_kernel(...):
    ...
```


### 3. Tuning for Vector Operators


Vector operators support the minimal tuning parameters:


```python
@max_autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 1024}),
        triton.Config(kwargs={'BLOCK_SIZE': 2048}),
    ],
    key=["N"],
    kernel_type="vector",
    num_stages=[1, 2],
    enable_ubuf_saving=[True, False],
)
@triton.jit
def vector_kernel(...):
    ...
```


## Summary


`max_autotune` is an advanced automatic tuning tool provided by Triton-Ascend, suitable for the following scenarios:


1. Given the tiling parameter space, the goal is to reduce the manual effort of enumerating configurations.  
2. Joint tuning of multiple Ascend compilation parameters (such as `num_stages`, `enable_hivm_auto_cv_balance`, etc.) is required.  
3. The aim is to batch-generate candidate configurations via Cartesian product.


The core value of `max_autotune` lies in: using a small amount of base configuration plus a description of the tuning parameter space to automatically expand into a complete set of candidate configurations, balancing flexibility and convenience.

