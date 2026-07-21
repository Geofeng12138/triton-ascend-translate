# Triton-Ascend autotune User Guide


## Document Positioning


This document is intended for users who already know how to write Triton kernels and understand the basic concepts of the community version of `triton.autotune`. It focuses on explaining the recommended usage of Triton-Ascend:


- Recommended autotune writing style on Triton-Ascend;  
- The meaning of `configs=[]` in the Ascend backend;  
- Applicable boundaries of the automatic Tiling mode, and when to fall back to handwritten `triton.Config`.


## Quick Start


On Triton-Ascend, it is recommended to retain the basic syntax of the community version `@triton.autotune`; when you want the system to automatically generate and evaluate candidate configurations, set `configs` to `[]`.


```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime


@triton.autotune(
    configs=[],
    key=["M", "N"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ...
```


This means:


- The semantics of `key` remain consistent with the community version, used to determine which input changes will trigger re-selection of configurations;  
- `configs=[]` in Triton-Ascend means "the Ascend backend automatically generates candidate configurations and completes optimization," rather than "no available configurations."


### 1. Enable Ascend's extension for autotune first


Only after importing the following line will the autotune extension path of Triton-Ascend be entered:


```python
import triton.backends.ascend.runtime
```


If this step is not performed, the community version of `triton.autotune` will still be used, and `configs=[]` will not trigger Ascend's automatic Tiling generation.


### 2. `@triton.autotune` must be directly wrapped around `@triton.jit`


Must be written in the following order:


```python
@triton.autotune(configs=[], key=["M", "N"])
@triton.jit
def kernel(...):
    ...
```


`@triton.autotune` must be directly wrapped around `@triton.jit`, with no other decorators inserted between them. Otherwise, the kernel DSL cannot be parsed, preventing entry into Triton-Ascend's automatic tiling generation and optimization pipeline.


### 3. The meaning of `key` is consistent with the community


The essence of `key` is the cache key for autotune. Any parameter filled into `key` will trigger a re-autotune whenever its value changes.


In most cases, `key` contains shape parameters such as `M/N/K`, `seq_len`, and `hidden_size`, as these often significantly affect the optimal Tiling. However, `key` is not limited to shape parameters; any parameter whose variation influences configuration selection can also be placed in `key`.


### 4. Parameters intended for automatic tuning should not be fixed in advance


If you want a `tl.constexpr` to participate in automatic Tiling generation, the following three conditions must be met simultaneously:


- It must itself be a Tiling parameter, meaning it affects the data size or tile size processed by each block (logical core);  
- Do not explicitly pass a fixed value for it at launch;  
- Do not set a default value for it in the kernel definition.


For example, in the following code, `BLOCK_M` will participate in auto-tuning:


```python
kernel[grid](
    x,
    y,
    out,
    M,
    N,
)
```


If you explicitly pass in during launch:


```python
kernel[grid](
    x,
    y,
    out,
    M,
    N,
    BLOCK_M=128,
)
```


Then this parameter has been fixed and is no longer within the scope of automatic generation.


Similarly, if a default value is provided for a tuning parameter in the kernel definition, for example:


```python
@triton.jit
def kernel(
    ...,
    BLOCK_M: tl.constexpr = 128,
):
    ...
```


Then this parameter will not participate in automatic tuning. For parameters that are intended to be automatically generated and optimized by the framework, they should be kept as `tl.constexpr` that are "not explicitly assigned a value at launch and have no default value in the kernel definition."


### 5. If Tiling parameters affect the grid, the grid must be written in lambda form


If a Tiling parameter affects the grid size, the grid cannot be pre-defined as a fixed value or a static expression that only depends on runtime parameters. Instead, it must be written as a `lambda` form that depends on meta parameters. This is consistent with the requirements of the community autotune.


```python
grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
```


The reason is that when autotune evaluates different candidate configurations, the values of parameters such as `BLOCK_M` will change; if the grid does not change along with the candidate configuration, it cannot be guaranteed that each candidate configuration is executed with the correct launch method.


## Usage Precautions


The benchmark semantics of autotune are consistent with the community, executing the kernel multiple times. If the kernel has side effects, such as containing atomic operations, inplace writes, or modifying the cumulative state of input/output buffers, it still needs to be handled through the existing hook mechanism in the community.


## Triton-Ascend Extensions Compared to Community Autotune


The typical pattern of the community edition `triton.autotune` is: users manually provide a set of `triton.Config`, the framework performs benchmarking, and then caches the optimal result.


Triton-Ascend mainly extends the following aspects while keeping the interface conventions unchanged.


### 1. Support `configs=[]` for automatic generation of candidate configurations


This is the most core extension. Users do not need to manually write a set of `triton.Config` first; instead, they can leave `configs` empty and let the Ascend backend automatically generate candidate configurations based on the kernel DSL semantics and runtime shapes.


### 2. Parallel Compilation Supporting Multiple Configs


When autotune needs to evaluate multiple candidate configurations, Triton-Ascend will compile these candidate configurations in parallel by default to reduce the first-time tuning latency.


This capability is enabled by default and can be disabled via the environment variable `TRITON_AUTOTUNE_PARALLEL_COMPILE=0`.


### 3. Support using profiler to collect kernel performance


Triton-Ascend supports switching performance collection methods during autotune benchmarking: in addition to the default benchmark mode, you can also use the profiler to collect kernel performance data for each candidate config. It focuses only on the on-chip computation time of the kernel, which is more accurate than the default performance collection method for kernels with short execution times, but it adds some overhead. This capability can be enabled via the environment variable `TRITON_BENCH_METHOD='npu'`.


## Scope and Behavior of Automatic Tiling Generation


The first point above explains that Triton-Ascend supports `configs=[]` for automatically generating candidate configurations. The following points provide further explanation of this capability.


### 1. Auto-generation focuses on Tiling parameters


The focus of Ascend's automatic generation is on the `tl.constexpr` parameters related to Tiling in the kernel, that is, the parameters that affect the data size or tile size processed by each block (logical core).


This capability is not equivalent to "automatically tuning all parameters for you." Compilation parameters such as `num_warps` and `num_stages`, as well as non-Tiling parameters of the kernel, are not within the current scope of automatic generation.


### 2. Candidate configurations will include Ascend hardware constraints


When generating candidates, the Ascend backend filters them based on constraints such as NPU on-chip memory capacity, alignment requirements, and core utilization, rather than simply enumerating a set of configurations and blindly benchmarking them.


### 3. The goal of the automatic tiling mode is to conveniently provide configurations with "decent performance."


To balance optimization time and optimization effectiveness, the current automatic Tiling performs extensive pruning on the number of generated configurations, so it does not guarantee that the automatically generated results will necessarily reach the performance upper bound of manual extreme tuning.


The goal of this capability is to conveniently provide users with a reasonably good Tiling configuration while minimizing the user's learning curve and optimization cost.


### 4. When automatic tiling generation fails, users need to manually write `triton.Config`


If the automatic Tiling mode fails to generate any usable candidate configurations, the user needs to manually write `triton.Config` instead. It is also recommended to submit an issue for such scenarios to help Triton-Ascend improve its parsing and automatic generation capabilities in the future.


## Handwritten `triton.Config` Mode


If the automatic Tiling mode fails to generate, or the generated Tiling performance does not meet expectations, simply revert to the standard community writing style. Triton-Ascend maintains compatibility with this part of the semantics.


```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}),
    ],
    key=["M", "N"],
)
@triton.jit
def kernel(...):
    ...
```


In this mode:


- The configuration is manually provided by the user;  
- The framework is responsible for benchmarking, selecting the optimal configuration, and caching for reuse;  
- The usage habits are consistent with the community autotune.


## Advanced Usage: Joint Tuning of Automatic Tiling with Other Parameters


The following content pertains to advanced usage. It should only be considered when the user wishes to further jointly tune non-tiling parameters or compilation parameters of the kernel while in automatic tiling mode.


### 1. Community autotune: Manually enumerate all tuning parameters


The standard community practice is for users to manually enumerate all candidate configurations.


```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": GS,
            },
            num_warps=num_warps,
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BK in [16, 32, 64]
        for GS in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...
```


### 2. Triton-Ascend: Manual Enumeration of Tiling Parameters and Ascend Compilation Parameters


In Triton-Ascend, if you wish to continue using the manual enumeration mode, you can also include the Ascend-side parameters together in the manual configuration space.


```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": GS,
                "multibuffer": MS,
            }
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BK in [16, 32, 64]
        for GS in [1, 2, 4, 8]
        for MS in [False, True]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...
```


### 3. Triton-Ascend: Automatically Generate Tiling While Jointly Tuning Other Parameters


If you want the Tiling parameters to continue being automatically generated by `configs=[]`, but also wish to tune other non-Tiling parameters or compilation parameters at the same time, you can pass these additional search dimensions via `hints`:


```python
@triton.autotune(
    configs=[],
    key=["M", "N", "K"],
    hints={
        "GROUP_SIZE_M": [1, 2, 4, 8],
        "multibuffer": [False, True],
    },
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...


matmul_kernel[grid](a, b, M, N, K)
```


The meaning of this approach is:


- Tiling-related parameters are still automatically generated by Triton-Ascend;  
- Non-Tiling parameters or compilation parameters are explicitly provided by the user as candidate sets via `hints`;  
- Autotune evaluates the configuration space formed by the combination of these two parts.


## Summary


Triton-Ascend's key extension compared to the community autotune is not to change the user interface, but to add the ability to "automatically generate tiling candidates and complete optimization" on top of the community interface. For most users, the most recommended usage is:


- Keep the writing style of the community edition `@triton.autotune`;
- Set `configs` to `[]`;
- Enable the Ascend backend to automatically perform candidate generation, selection, benchmarking, and cache reuse based on the kernel DSL and runtime shapes.


If the scenario is not suitable for the automatic Tiling mode, simply revert to manually writing `triton.Config`.

