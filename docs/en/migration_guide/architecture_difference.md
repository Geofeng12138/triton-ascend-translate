# Development Differences Between Ascend and GPU


## Multi-Core Task Parallelization Strategy


NPU in Triton multi-core parallelism adopts a physical core hard-binding mode, which forms a core difference from the GPU's logical dimension parallelism plus hardware automatic physical mapping mode. The core comparison is shown in the table below:


| Dimension | GPU (NVIDIA) | Ascend |
|-----------|--------------|--------|
| Grid Essence | Logical task dimension (decoupled from physical cores) | Physical core group mapping (bound to AI Core topology) |
| Core Count / Dimension Limit | No hard limit on grid dimension/size | Grid size ≤ total AI Cores, 2D must match topology |


GPU can bind multiple dimension axes (a three-dimensional grid=[n,m,l] is equivalent to n×m×l parallel threads), where each thread corresponds to exactly one kernel execution and executes only once.  
NPU's Vector core and Cube core are multiple physical cores, with different numbers of cores across hardware generations. Each core executes only one Block and supports repeated scheduling and execution of that Block.


### Fully Utilize Core Count


Ascend NPU has multiple compute cores. Reasonably allocating and fully utilizing all available cores is one of the key factors in improving operator performance.
When calling a Triton kernel function, the number of cores used is controlled by setting the launch parameters. Taking the GELU operator as an example:


```Python
triton_gelu[n, 1, 1](...)  # 第一个参数表示使用的核数，n表示使用n个核
```


By tuning the number of cores, full scheduling and utilization of all computing resources can be achieved, thereby maximizing parallelism and throughput. When `auto-blockify` is not enabled (see the next section), the number of cores for launching a grid must be less than or equal to 65,535.


### auto-blockify: Breaking the 65,535 Logical Block Limit


Community Triton treats the grid as a purely logical dimension on NVIDIA GPUs — `n` logical blocks are mapped 1:1 to `n` hardware blocks, distributed to SMs by the hardware at runtime, and each block does not require internal looping. On Ascend, due to the strong physical core binding described in the previous section, the upper limit of launchable grids is capped at 65,535, which is too restrictive for kernels containing millions of logical work items (such as autotuned reduce/scan, megablocks-style sparse kernels, etc.).


`auto-blockify` (the `SIMTAutoBlockify` compile-time pass + the corresponding runtime cap) eliminates this limitation by "treating it as logical at compile time and folding it into physical cores at startup."


- **Compile time**: The Triton pass wraps the kernel function body inside a `scf.for` loop, with the iteration variable provided by `gpu.linear_block_id`. Chunk size = `ceildiv(logical_block_count, physical_core_count)`, each physical block sequentially runs `chunk` logical block IDs.
- **Runtime**: The block-count parameter passed to the launcher is clamped from the logical grid to `physical_core_count`, consistent with the folding at compile time.


Both sides share the same gating metadata (`enable_auto_blockify` on `NPUOptions`, falling back to `TRITON_ALL_BLOCKS_PARALLEL` when not passed). The compile-time loop wrapping and runtime cap are always synchronized — there is no scenario where a kernel is compiled under one mode but launched under another.


Notes for porting from GPU Triton kernel:


- Grids larger than 65,535 can run directly without manually folding the outer dimension into the kernel function body.
- Logical blocks must be order-independent (loops access chunks in order). Kernels that rely on strict logical block ID order (e.g., cross-block synchronization based on a specific order) need to be rewritten.
- Per-block workspace allocation is reduced from `O(logical_block_count)` to `O(physical_core_count)` because the workspace is reused across iterations of the inner `scf.for`.


## Single-Core Data Movement Strategy


### Data Tiling


When writing Triton kernel functions, a reasonable data partitioning strategy is crucial for performance optimization. By adjusting different granularity parameters, the computational load and memory access efficiency can be balanced across various dimensions.


Common segmentation parameters include:


```text
ncore：使用的核数（跨核切分）
xblock：核间数据块大小（核间切分）
xblock_sub：核内切分粒度（核内细粒度划分）
```


Developers can manually select the optimal sharding configuration based on the actual scenario, making full use of on-chip memory for each computation as much as possible, thereby avoiding performance bottlenecks caused by frequent access to global memory.


Taking the GELU operator as an example, by adjusting the tiling parameters, it can effectively adapt to the on-chip cache capacity constraints, thereby improving execution efficiency.


Note: The on-chip memory capacity of the Atlas 800T/I A2 product is 192KB. Therefore, this limitation must be considered when designing the partitioning strategy to ensure that the data volume per computation round does not exceed the on-chip memory capacity.


#### GELU Operator Example


GELU operator development example, using three methods to compute the result.


standard_unary is for standard Torch computation.


`triton_easy_kernel` is a simple Triton implementation.


The `triton_better_kernel` is a more efficient Triton implementation.


#### Standard Torch Approach


Input tensor x0, compute the GELU operator via torch, and return the resulting value.


```Python
def standard_unary(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res
```


#### Simple Triton Implementation


Below is a simple kernel example written in Triton, demonstrating how to define and invoke a basic Triton kernel function. This example implements a simple mathematical operation (the GELU activation function).


```Python
# 定义triton_kernel核函数
@triton.jit
def triton_easy_kernel(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr0 + idx_block, ret)
```


Notes


1. Memory limitation: In the above implementation, all input data is loaded into memory at once for computation. If the input tensor is too large, it may exceed the on-chip memory capacity of a single kernel, leading to out-of-memory errors. Therefore, this simple implementation is more suitable for small-scale tensor computations or for understanding the basic syntax and invocation methods of Triton kernels.


2. Applicable Scenarios: Although this approach helps quickly understand and get started with Triton programming, for large-scale datasets or applications with high-performance requirements, it is recommended to adopt more complex data partitioning strategies (such as Tiling) to fully utilize hardware resources and avoid memory overflow issues. In this way, developers can quickly get started with Triton programming while learning how to define, invoke, and optimize Triton kernel functions.


#### More Efficient Triton Implementation


When writing high-performance operators using Triton on Ascend NPU, a data tiling strategy is typically required to fully utilize hardware resources, avoid memory overflow, and improve execution efficiency. Below is an example of an optimized Triton kernel implementation suitable for large-scale tensor computations.


```Python
# 定义triton_kernel核函数
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

# 调用triton_kernel核函数
ncore = 32
xblock = 32768
xblock_sub = 8192
triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
```


Key Code Explanation


```Python
# 计算当前核处理数据块的起始偏移地址，实现核间切分。每个核仅负责 XBLOCK 大小的数据范围。
xoffset = tl.program_id(0) * XBLOCK

# 在单个核内部进一步细分数据块，每次处理 XBLOCK_SUB 大小的数据，实现核内切分。
for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):

# 构造当前迭代的数据索引数组，用于访问输入和输出张量。
x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

# 设置掩码以防止越界访问，确保只处理合法范围内的数据。
xmask = x_index < xnumel

# 分别用于从全局内存加载数据到片上内存，以及将计算结果写回全局内存。
tl.load() 和 tl.store()
```


## Compilation Optimization Capabilities


### AscendNPU IR Optimization


For the Ascend hardware and software features, the compilation options optimized for AscendNPU IR have been adapted, as shown in the table below.  
**Usage**: During the autotune configuration phase, pass in the value of the compilation option.  
Taking enabling the `multibuffer` option as an example, during the autotune configuration phase, i.e., in `triton.Config`, pass `'multibuffer': True`. For details, see the [autotune example](../examples/06_autotune_example.md).


```python
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),]
```


| Option | Capability | Enabled |
| ----------------- | ------------ | ----------------- |
| multibuffer | Enables pipeline parallel data transfer | Default true; true, false. Configurable in autotune |
| unit_flag | An optimization item for cube data transfer | Default None; true, false. Configurable in autotune |
| limit_auto_multi_buffer_only_for_local_buffer | An optimization item for CV operators and cube data transfer | Default None; true, false. Configurable in autotune |
| limit_auto_multi_buffer_of_local_buffer | Specifies the scope for enabling double buffer in cube operators | Default None; possible values "no-limit" or "no-l0c", configurable in autotune |
| set_workspace_multibuffer | Configures the workspace multi-buffer level to enable multi-buffering for workspace-related data transfer. | Default None; can take a single value, e.g., 2 or 4; configurable candidate values in autotune |
| enable_hivm_auto_cv_balance | Enables or disables automatic CV balance to balance Cube and Vector execution in CV fusion scenarios. | Default None; true, false. Configurable in autotune |
| tile_mix_vector_loop | An optimization item for CV operators, indicating how many parts the current vector can be split into | Default None; can take a single value, e.g., 2, 4, or 8; configurable candidate values in autotune |
| tile_mix_cube_loop | An optimization item for CV operators, indicating how many parts the current cube can be split into | Default None; can take a single value, e.g., 2, 4, or 8; configurable candidate values in autotune |
| auto_blockify_size | An optimization item for TRITON_ALL_BLOCKS_PARALLEL, used to specify the size of the first dimension extended from the left. | Default 1; can take a single integer value, e.g., 2, 4, or 8; configurable candidate values in autotune |
| enable_auto_blockify | Overrides the `TRITON_ALL_BLOCKS_PARALLEL` environment variable at the per-kernel level. When explicitly set to **true** or **false**, the kernel takes effect according to this value (ignoring the environment variable); when not set (None), it is determined by the environment variable. Priority: this option > environment variable > off. Both the compile-time blockify pass and the runtime block-count cap take effect based on this resolved value, and they are always consistent. | Default None; possible values **true** / **false** / None. |


- Note: The optimization compilation options are in the `ascend/backend/compiler.py` code.
- Note: The CV operator indicates that the operator uses both AI Core and Vector Core during its computation.

