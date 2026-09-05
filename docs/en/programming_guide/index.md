# Triton Operator Development Guide


Overview: This article focuses on notable issues in developing Triton operators on NPUs, covering three aspects: multi-core task parallelism, single-core data movement, and single-core data computation. First, in multi-core task parallelism, the basis for setting the maximum number of hardware cores and its specific implementation are introduced. Then, in single-core data movement, it details how to set an appropriate data block size within loops, describes commonly used optimization techniques during the process, and supplements the handling of potential UB OVERFLOW issues. Finally, returning to individual operators, it emphasizes how to develop Triton operators at the single-core data computation level and highlights related key points.


## Document Organization


This guide separates the general development principles from the operator development paths categorized by hardware execution units:


This page introduces common issues that all Triton-Ascend operators need to pay attention to, including kernel splitting, on-chip memory, memory access, Tiling, and Autotune.  
- [Vector Operator Development](./vector_operator.md) introduces operators such as element-wise, reduction, Gather/Scatter, etc., which are mainly executed by the Vector Core.  
- [Cube Operator Development](./cube_operator.md) introduces operators centered around `tl.dot`, matrix multiplication, and batch matrix multiplication.  
- [CV Fusion Operator Development](./cv_fusion_operator.md) introduces scenarios where Cube computation and Vector post-processing, reduction, Softmax, or cross-core collaboration coexist within the same operator.


For simple operators, refer to `docs/zh/examples/` and `third_party/ascend/tutorials/` in this repository first; for complex operators, refer to the complete optimization cases in `tutorial/best_practice/` of [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) on GitHub.


## General Multi-Core Task Parallelism


### Setting the Maximum Number of Hardware Cores


In a Triton operator, grid-based kernel partitioning is commonly used. For GPUs, the number of compute cores (SMs) typically ranges from tens to hundreds. However, for the Ascend NPU platform, the number of AI Core compute units is on the order of tens.  
Although the runtime interface allows a maximum of 65,535 concurrent tasks to be dispatched, tasks exceeding the number of physical cores are completed through a new round of dispatch. If Triton operators designed for GPUs are directly run on the Ascend platform, these numerous tasks will introduce considerable overhead from kernel launch and initialization, negatively impacting operator performance.  
Therefore, the kernel partitioning logic must be adapted to the characteristics of the Ascend platform. The most recommended approach is to **fix the number of kernel partitions to the physical core count of the hardware** and perform more fine-grained data partitioning within each core.


* For pure Vector operators, the number of sub-cores equals the **number of Vector cores**.
* For CV fusion operators, the number of sub-cores equals the **number of Cube cores** (typically half the number of Vector cores). During operator execution, Vector cores are invoked in a 1:2 ratio.


Generally, on an NPU card, a computing core AI Core contains one cube core, and each cube core is equipped with two vector cores. Therefore, the number of **vector cores (vectorcore_num)** and **cube cores (aicore_num)** can be obtained through the following interfaces:


```python
import torch
import triton.runtime.driver as driver
import torch_npu

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

```


Refer to the example code, first fix the number of cores, then process task chunks in batches through an inner loop.


```python
NUM_CORE = vectorcore_num
grid = (NUM_CORE ,)
_attn_fwd[grid](Q, K, V, M, Out, acc, scale, ...)

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, scale,
              ...,
              stride_qz, stride_qh,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # 计算任务总量,将三维任务(Z,H,M)展平为一维总任务数
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # 每个核根据自己标识选取要处理的任务
    pid = tl.program_id(0)  # 当前核的唯一ID
    NUM_CORE = tl.num_programs(0)  # 获取固定启动的总核数
    # 循环规则：range(pid, NUM_BLOCKS, NUM_CORE) 实现"跨步分配任务"
    # - 起始值pid：每个核从自己的ID开始取任务，避免任务重叠
    # - 步长NUM_CORE：按总核数跨步，确保任务均匀分配到各个核
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # 计算每次任务的数据偏移
        # 【核心：一维任务索引反向还原为原始多维索引】
        # block_idx是展平后的一维任务索引，通过整除/取余拆分回原始维度
        # 1.拆分Z+H合并轴 & M分块轴：
        #   - 整除NUM_BLOCKS_M：提取Z+H合并轴的索引（task_hz_idx）
        #   - 取余NUM_BLOCKS_M：提取M维度的分块索引（task_m_idx）
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        # 2.拆分Z+H合并轴为原始Z轴和H轴：
        #   - 整除H：还原Z轴索引（off_z）
        #   - 取余H：还原H轴索引（off_h）
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        # 3.计算数据偏移量：根据还原的Z/H索引，定位Q/K/V张量中对应的数据起始位置
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
```


## General Single-Core Data Transfer


### Setting Appropriate Block Size for Data Chunking Within Loops


Taking `add_kernel` as an example, variables and operations together determine the large on-chip memory space usage. By modifying the `BLOCK_SIZE`, you can adjust the size of data blocks within the loop and the intermediate computation results. If the limit is exceeded, the operator compilation will indicate the expected usage size and report an error. To achieve the maximum compute-to-memory-access ratio, `BLOCK_SIZE` should be as large as possible without exceeding the on-chip space. This can be done by pre-setting different `BLOCK_SIZE` values using Triton-Ascend's [Autotune](../examples/06_autotune_example.md), and the runtime will automatically select the optimal setting.


```python
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               out_ptr,
               n,  # 元素总数量。
               BLOCK_SIZE: tl.constexpr,  # 分块元素数量。
               ):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        # 分块大小为 BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        # 加载 x,y 数据到片上内存
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(out_ptr + offsets, output, mask=mask)
```


### Ensure that the size of the last axis of the tensor is aligned with the data


【Description】When VV-type operators need to invoke the Vector core for computation, the UB of Ascend hardware requires that the size of the last axis of the Tensor be divisible by 32 Bytes. For CV-type operators that need to invoke both the Vector core and the Cube core for computation, the size of the last axis of the Tensor must be divisible by 512 Bytes. If the last axis size is insufficient, it will be automatically padded. Under this premise, various operations on Tensors with shapes (2048,3) and (2048,1) in the model will suffer significant performance degradation due to automatic padding. In such cases, consider using a transpose operation to move the aligned axis to a lower dimension, and then transpose back to the original state when storing, thereby avoiding automatic padding and optimizing computation speed. At the same time, since the transpose operation itself is also affected by the automatic padding rules, special techniques are also needed to avoid padding. Here is a tip called "borrowing axis transpose," applicable to the scenario where **tensor.numel() % 256Byte == 0**. The specific operation is as follows:


- Note: VV-type operators indicate that only the Vector Core is used during computation; CV-type operators indicate that both the AI Core and the Vector Core are used during computation.  
- Example


```python
# conv_state = tensor([2048, 3], bfloat16)
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3)) # 当成1D tensor load，此时由于numel对齐，不会自动补齐。
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,) # 长轴(2048)裂出一根对齐轴(16)借给短轴(3)，从而让两个轴都对齐
```


### First, move the data to UB, then select the target value from UB.


[Description] In the discrete scenario of NPU, data can first be moved to UB, and then the target value can be selected from the shared memory.


- Example


```diff
@triton.jit
def pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # 原先写法
    # val = tl.load(x_ptr + idx * stride_x, mask=mask)
    # 修改后写法
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```


- Performance analysis and comparison before and after optimization


By executing the test case using the `msprof` tool, you can obtain the `PROF_*` folder, which contains the `op_summary_*.csv` file. This file helps analyze the pipeline status. Note: "*" represents a timestamp. [Reference method for performance data collection](../debug_guide/profiling.md).


||Op Name|aiv_mte2_time(us)|aiv_mte2_ratio|
|:---- |:--------|:--------|:--------|
|Unoptimized|pick_kernel|0.686|0.008|
|Optimized|pick_kernel|1.041|0.066|


By analyzing the data in the table, it can be observed that there is a significant difference in `aiv_mte2_time(us)` and `aiv_mte2_ratio` before and after optimization. The optimization approach first moves most of the data to UB, reducing the number of times small batches of data are transferred to UB via L2, thereby decreasing the total time for L2-to-UB transfers.


### Storage-Compute Parallelism


Triton-Ascend supports two data processing modes: storage-compute serial and storage-compute parallel.


Storage-compute serialization: Data is first transferred from global memory to on-chip memory, and after computation is completed, the next batch of data is transferred. This approach has significant idle waiting time and low efficiency.


Memory-computation parallelism: While transferring the first batch of data to on-chip memory, computation on it has already begun; subsequently, the second batch of data is transferred, forming a pipelined operation where "data transfer + computation" overlap, significantly improving overall throughput.


The key to achieving memory-computation parallelism lies in properly designing the data tiling strategy, so that during the computation of the current batch of data, the data required for the next stage can be prepared in advance, thereby enabling parallelization of data movement and computation. Currently, the compiler defaults to configuring multiBuffer=True, which supports memory-computation parallelism by default.


### Tiling Optimization


When AI Core performs computation, data must first be moved to the on-chip memory. The on-chip memory space is usually much smaller than the total amount of data to be processed by the AI Core. Taking the Atlas 800T/I A2 product as an example, the on-chip memory capacity is 192KB. After the double buffer feature is enabled by default, the capacity is further reduced to half of the original. Therefore, during operator computation, data needs to be partitioned into blocks, and only a small portion of the data is loaded and processed at a time.


- Example


```diff
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
- def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
+ def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
+   base_offset = pid * BLOCK_SIZE

+   # 计算需要处理的块的总数
+   num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

+   # 针对每个子块进行循环处理
+   for sub_block_idx in range(num_sub_blocks):
+       # 计算当前子块的偏移量
+       sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
+       offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
-       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        # Load input and mask
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # Write the original input first
        tl.store(out + offsets, input_vals, mask=mask)

        # Overlay and write value at the position that needs to be filled
-       value_to_write = tl.full([BLOCK_SIZE], value, dtype=input_vals.dtype)
+       value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0))
        tl.store(out + offsets, overwrite_vals, mask=mask)
```


### Triton Autotune


In Tiling optimization, the values of block parameters such as BLOCK_SIZE and BLOCK_SIZE_SUB directly affect operator performance, but manually tuning parameter combinations is inefficient and makes it difficult to find the optimal values. triton.autotune is an automatic tuning tool provided by the Triton framework that can iterate through preset parameter configurations, compare performance through actual execution, and automatically select the optimal parameter combination, making it a core supporting method for Tiling optimization.


If you are interested in the recommended usage of `configs=[]` on Triton-Ascend and the applicable boundaries of automatic Tiling, please refer to the [Triton-Ascend autotune usage guide](../autotune_guide.md).


- Core Function  
Automatically traverse the parameter space: For constexpr type block parameters such as BLOCK_SIZE and BLOCK_SIZE_SUB, batch test the performance of different values.  
Performance benchmark comparison: Use the operator execution time as the metric to select the optimal parameters adapted to the current hardware.  
Cache tuning results: The optimal configuration after tuning is cached, and subsequent operator calls directly reuse it to avoid repeated tuning.


- Simple Example


    ```diff
    import triton.language as tl

    @triton.autotune(
    configs=[ # 待测试的参数配置列表,参数候选值需要是2的幂次
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
        ],
        key=['n_elements'], # 调优维度：参数取值依赖的输入维度
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    ```


- Note: Set the following environment variables to print the optimal parameter information.


    ```diff
    export TRITON_PRINT_AUTOTUNING=1
    ```


### Advanced: Using max_autotune for Automatic Tuning


For Ascend NPU operators, to achieve optimal performance, in addition to `BLOCK_SIZE`, multiple hardware-related parameters such as `num_stages`, `enable_hivm_auto_cv_balance`, and `tile_mix_vector_loop` also need to be tuned. Manually enumerating all combinations using `@triton.autotune` would cause the configuration list to explode, making the code difficult to maintain.


max_autotune is an extended decorator specifically designed for Ascend NPU (located in triton.backends.ascend.runtime), which allows users to provide only the basic configuration while passing the remaining tuning parameters as a list. The decorator automatically generates a Config list for all combinations.


- Core Function  
Developers only need to provide a small amount of basic configuration (such as `BLOCK_SIZE`), and all compiler options related to the operator type (e.g., `num_stages`, `enable_hivm_auto_cv_balance`, `tile_mix_vector_loop`, `enable_ubuf_saving`, etc.) will be automatically included in the search for the optimal combination through built-in reasonable default values. This eliminates the need for developers to explicitly enumerate options, enabling one-shot automatic optimization of the best tiling and compiler option combination. If developers wish to restrict certain parameters, they can also override the default search range by explicitly passing a list.


- Simple Example


    ```diff
    from triton.backends.ascend.runtime import max_autotune

    @max_autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
        ],
        key=['n_elements'],
        kernel_type="vector",           # 算子类型，支持 cube/mix/vector
        enable_ubuf_saving=[True, False] # 可选，默认已包含
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, **META):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    ```


### How to Avoid UB OVERFLOW on NPU


[Description] On the NPU, there is an upper limit for UB or L1 Size. When this error occurs, it is necessary to reduce the amount of data transferred in a single operation and handle long sequence scenarios using a for loop.


```diff
E triton.compiler. errors.MLIRCompilationError:
E ///--------------------- [ERROR][Triton][BEG]-------------------------
E [ConvertLinalgRToBinary] encounters error:
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":2:1): error: Failed to run BishengHIR pipeline
E
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":3:3): error: ub overflow, requires 3072256 bits while 1572864 bits available! (possible reason
large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer. )
```


[Note] The UB size of A2 series products is 192KB (1572864 bits).


## General Single-Core Data Operation


### Development Goals


Implement basic data operation operators (such as addition, subtraction, multiplication, division, activation functions, and simple matrix element operations) on a single Ascend NPU core. Ensure efficient execution of operators within a single core, laying the foundation for subsequent multi-core parallelism and distributed scaling.


### Development Steps


1. Determine operator functionality  
- Clarify the shape and data type (float16/float32/int32, etc.) of input/output tensors.  
- Confirm whether broadcasting or boundary handling is required.


2. Writing the kernel function  
Single-kernel operations typically correspond to block-level data processing.  
Example of single-kernel data operation: vector addition


```diff

@triton.jit
def add_kernel(x_ptr, # Pointer to first input vector.
    y_ptr, # Pointer to second input vector.
    output_ptr, # output 向量的指针.
    n_elements, # 向量的大小.
    BLOCK_SIZE: tl.constexpr, # 每个进程需要处理的元素个数.
    # 注意：constexpr属性表示它可以被用作shape值.
):
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```


Call:


 ```diff
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```


Use the above function to compute the element-wise sum of two torch.tensor objects and test its correctness.


 ```diff
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
f'{torch.max(torch.abs(output_torch - output_triton))}')
# Out:
# tensor([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], device='npu')
# tensor([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], device='npu')
# The maximum difference between torch and triton is 0.0
```


3. Key points of single-core computing


Block-level data processing: Each compute block is responsible for a small segment of data, ensuring parallelism.


- Boundary check: Use mask or `if (tid < N)` to avoid out-of-bounds access.


-Block size selection: Reasonably set block and grid


4. Performance Key Points:  
(1) Memory Access Optimization  
- Ensure sequential access.  
- Use aligned strides to avoid cross-row/cross-column jump access.  
- Align data block sizes to 32-byte boundaries as much as possible.  
Ensure input and output buffers are aligned during allocation to avoid memory access performance degradation.  
Example:


 ```diff
BLOCK_SIZE = 256  # 256 * 4 bytes = 1024 bytes，对齐良好

@triton.jit
def vec_add_kernel(X, Y, Z, N,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # 计算当前 block 负责的 index 范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # mask 防止越界
    mask = offsets < N

    # 连续访存：offsets 是连续的
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)

    z = x + y

    # 连续写回
    tl.store(Z + offsets, z, mask=mask)


def vec_add(x, y):
    assert x.numel() == y.numel()
    N = x.numel()

    # 分配对齐内存（PyTorch 默认已经对齐到 64 字节）
    z = torch.empty_like(x)

    # grid：每个 block 处理 BLOCK_SIZE 个元素
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    vec_add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

    return z
```


(2) Sub-block Partitioning  
- Decompose the large matrix into smaller blocks, with each block completing computation within the UB.  
- Sub-block partitioning must balance memory access continuity and compute unit utilization.  
Example:


 ```diff
BLOCK_M = 64   # 每个 block 处理 64 行
BLOCK_N = 64   # 每个 block 处理 64 列
BLOCK_K = 32   # 内部累加维度

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # block 在 M 方向的 id
    pid_n = tl.program_id(1)  # block 在 N 方向的 id

    # 当前 block 对应的起始坐标
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 循环分块计算
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    # 写回结果
    c = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```



## General Multidimensional Tensor Slicing


When Triton operators handle multi-dimensional tensors, the core idea is to map high-dimensional data to hardware blocks, cores, and hardware units. This section provides typical processing examples for two-dimensional and three-dimensional tensors.


### 2D Tensor Partitioning: Taking Matrix Multiplication (GEMM) as an Example


For two-dimensional matrix multiplication, it is usually necessary to perform two-dimensional partitioning along the height (M) and width (N), and iterate in loops along the depth (K).


```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 1. 任务划分：计算当前 Block 在 M 和 N 维度上的坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 定义块指针（Block Pointers），处理多维步长（Strides）
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # 3. 循环迭代 K 维度进行累加计算
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N))
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, accumulator)
```


**Key Points**:


- `pid_m` / `pid_n` correspond to the block indices along the M / N dimensions, respectively.


- `stride_*` explicitly handles multi-dimensional strides, avoiding assumptions about contiguous memory.


- K dimension accumulates through loop tiling.


### Slicing of 3D and Higher-Dimensional Tensors: Taking Batched GEMM as an Example


When processing a 3D tensor (such as `[Batch, M, N]`), the `Batch` dimension (B) can be directly mapped to Triton's `Grid` dimension, or flattened with the `M/N` dimension and then remapped.


#### Add `Batch` dimension when starting `Grid`


```python
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']), B)
```


#### Kernel Function Implementation


```python
@triton.jit
def batched_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, B, ...):
    # 获取当前 Batch 的索引
    pid_b = tl.program_id(2)

    # 根据 Batch 索引计算全局内存的基地址偏移
    a_batch_ptr = a_ptr + pid_b * M * K
    b_batch_ptr = b_ptr + pid_b * K * N
    c_batch_ptr = c_ptr + pid_b * M * N

    # 后续 M、N、K 维度的切分与二维 GEMM 完全一致，只需替换基地址指针即可
    # ...
```


**Key Points**:


- `tl.program_id(2)` retrieves the index of the Batch dimension


- Each Batch independently calculates its own `a_batch_ptr` / `b_batch_ptr` / `c_batch_ptr`


The subsequent splitting logic for the M / N / K dimensions is consistent with that of the 2D GEMM.

