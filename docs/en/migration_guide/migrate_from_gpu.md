# GPU Triton Operator Migration


Overview: This document describes the general processing approach and common issues when migrating GPU Triton operators to the Ascend NPU. During migration, it is recommended to first complete the replacement of device and runtime interfaces on the Python side, then check grid partitioning, memory access alignment, single-core computation, UB space, and coreDim constraints, and finally complete code modification and correctness verification with specific examples.


## General Migration Process


### Migrating Python-side Device and Runtime Interfaces


Before modifying the specific Triton kernel, first complete the device migration on the Python side.


1. Add `import torch_npu` in the Python file.
2. Locate device specifications such as `device="cuda"`, `device='cuda'`, `.cuda()`, and `.to("cuda")`, and change them to `device="npu"`, `device='npu'`, `.npu()`, or `.to("npu")`.
3. Locate GPU-specific interfaces such as `torch.cuda.*`, CUDA stream, CUDA event, and CUDA synchronize, and replace them with the corresponding NPU interfaces or remove unnecessary synchronization logic.
4. Remove logic that serves only GPU device discovery, such as device assertions related to `triton.runtime.driver.active.get_active_torch_device()`.
5. Keep the main logic of the Triton kernel unchanged, and first use NPU tensors to complete compilation and correctness verification.


### Adjusting grid core allocation


A common practice on GPUs is to design the grid as a large number of logical programs, which are scheduled to SMs for execution by hardware and runtime. When migrating to NPUs, priority should be given to the number of physical AI Cores and operator types.


- Grid prefers 1D; 2D NPU adaptation writes will also be merged into 1D, for example, `(20,)` and `(4, 5)` have the same effect.  
- The number of concurrent tasks for vector-only operators is typically organized by the number of Vector Cores; operators containing `tl.dot` are usually organized by the number of AI Cores.  
- When the logical grid is much larger than the number of physical cores, it is necessary to evaluate whether to change to processing multiple tiles in a loop within each program, or use `TRITON_ALL_BLOCKS_PARALLEL` when there is no sequential dependency between logical cores.  
- coreDim cannot exceed `UINT16_MAX` (65535). For large-shape operators, the grid size needs to be controlled in combination with BLOCK_SIZE or the tiling method.


| Dimension | Core Architecture | Operator Type |
|-----------|------------------|--------------|
| Ascend NPU | Multiple AI Cores, divided into Cube Core (matrix multiplication) and Vector Core (vector computation) | Vector-only operators → number of concurrent tasks = number of Vector Cores; operators containing `tl.dot` → number of concurrent tasks = number of AI Cores |
| GPU NVIDIA/AMD | Multiple CUDA Cores (scalar/vector computation) + Tensor Cores (matrix multiplication) | For GPU operators, the degree of concurrency is generally determined automatically by the compiler and hardware |


### Check Single-Core Data Transfer


After completing the device replacement, it is necessary to continue checking the data transfer method within a single program.


- In the Vector operator scenario, 32-byte memory access alignment is required, while in the cube-vector fusion operator scenario, 512-byte alignment is required.
- Retain the tail mask to ensure that boundary elements do not cause out-of-bounds access.
- Check the on-chip memory usage of a single tile to avoid triggering UB space overflow.
- Remove or replace GPU-specific synchronization APIs, such as CUDA thread, stream, event, or kernel synchronize related interfaces.


### Check Single-Core Data Computation


NPUs and GPUs differ in their compute units and supported data types. After migration, correctness should be ensured first, followed by adjustments based on performance issues.


- For intermediate values such as integer indices, offsets, and lengths, first confirm whether the current data type is efficiently supported by the NPU path.
- For operators containing `tl.dot`, verify that the M/N/K tiles, accumulation dtype, and output dtype meet the NPU backend requirements.
- For long sequences, large hidden sizes, or large K-dimension loops, prioritize controlling the single load and computation scale through tiling.


## Migration Example


### Example 1: Complete Migration of Vector Addition


```diff
import torch
+import torch_npu  # 【新增】导入昇腾NPU PyTorch适配库，提供NPU设备支持
import triton
import triton.language as tl

-DEVICE = triton.runtime.driver.active.get_active_torch_device()  # 【删除】GPU设备自动获取，NPU无需此逻辑

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector.
    y_ptr,  # Pointer to second input vector.
    output_ptr,  # Pointer to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
-    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE  # 【删除】GPU设备一致性校验，NPU无需显式断言
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
-x = torch.rand(size, device='cuda')  # 【删除】GPU设备指定
+x = torch.rand(size, device='npu')  # 【修改】指定为昇腾NPU设备
-y = torch.rand(size, device='cuda')  # 【删除】GPU设备指定
+y = torch.rand(size, device='npu')  # 【修改】指定为昇腾NPU设备
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)
```


### Example 2: Device Replacement and Single-Core Data Transfer


The following example demonstrates correctness verification for a single-core data transfer scenario after replacing the device from CUDA with NPU:


```diff
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def fn_broadcast_1d(output_ptr, x_ptr, XS: tl.constexpr, YS: tl.constexpr):
    xidx = tl.arange(0, XS)[None, :]
    base = tl.load(x_ptr + xidx)
    out = base.broadcast_to((YS, XS))
    oidx = tl.arange(0, YS)[:, None] * XS + tl.arange(0, XS)[None, :]
    tl.store(output_ptr + oidx, out)

@pytest.mark.parametrize('shape', [(1,), (2,), (4,)])
@pytest.mark.parametrize('dtype', [torch.int32])
def test_npu_1d(shape, dtype):
    XS = shape[0]
    YS = 4

-    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device='cuda')
+    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device='npu')
    std = torch.broadcast_to(x, (YS, XS))
-    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device='cuda')
+    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device='npu')
    fn_broadcast_1d[(1,)](output, x, XS, YS)
    assert torch.allclose(std, output)
```


## Common Issues Overview


After completing the basic migration steps, you may encounter new issues, which can be summarized into the following two categories:


1. coreDim limitation issue


Triggered when the grid dimension exceeds the NPU hardware limit.  
Typical error message: `coreDim=xxxx can't be greater than UINT16_MAX`.


2. UB space overflow


Memory usage exceeds NPU cache capacity.  
Typical error message: `ub overflow, requires xxxx bits while 1572864 bits available!`.


### Resolving coreDim Exceeded Limit


Problem Analysis:  
The `coreDim` parameter of the NPU cannot exceed `UINT16_MAX` (65535). When processing large-scale data, simple grid partitioning may cause this limit to be exceeded.


Case: Optimization of the zeros_like function  
Data scale: N = 1073741824, original BLOCK_SIZE = 2048, calculated coreDim = 524288 > 65535 (exceeds limit)


Solution 1:  
The Ascend compiler provides a corresponding solution for the coreDim overflow issue. Simply set the environment variable `TRITON_ALL_BLOCKS_PARALLEL` to 1. The command to set it is as follows:


```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```


Solution 2:  
Increase `BLOCK_SIZE` to reduce the number of required cores, ensuring that `coreDim` does not exceed the limit.  
The calculation formula is as follows:


```text
coreDim = ceil(N / BLOCK_SIZE)
ceil(N / BLOCK_SIZE) <= 65535
BLOCK_SIZE >= ceil(N / 65535)
```


Substituting `N = 1073741824` yields:


```text
ceil(1073741824 / 65535) = 16385
triton.next_power_of_2(16385) = 32768
```


Therefore, if `BLOCK_SIZE` takes a power of 2, it should be set to at least `32768`.


Code before optimization:


```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)

def zeros_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device # x.device = "npu"
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)  # 原始值过小
    return out
```


Optimized code:


```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)

def zeros_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device # x.device = "npu"
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
    BLOCK_SIZE = max(32768, min_block_size) # 至少为 32768
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out
```


### Dynamically Calculate the Appropriate BLOCK_SIZE to Avoid Exceeding coreDim Limit


```diff
optimal_block_size = 32768  # 根据计算得出的优化值

grid_fn = lambda meta: (triton.cdiv(N, optimal_block_size),)

zeros_kernel[grid_fn](out, N, BLOCK_SIZE=optimal_block_size)
return out
```


### Handling Composite Issues: coreDim + UB Overflow


Problem Analysis:
In some cases, resolving the coreDim issue may lead to a new UB overflow problem. This typically occurs when, after increasing the BLOCK_SIZE, the amount of data that a single thread block needs to process exceeds the UB cache capacity of the NPU.


Case:
Data size: N = 1073741824, original BLOCK_SIZE = 4096, calculated coreDim = 262144 > 65535 (exceeds limit). After adjusting BLOCK_SIZE to 32768, coreDim = 32768 (compliant), but UB overflow occurs.


Solution:  
Introduce the `BLOCK_SIZE_SUB` parameter to further subdivide large blocks, maintaining a reasonable `coreDim` while controlling memory usage.  
Code before optimization:


```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(~fill_mask) & mask, other=0)
    tl.store(out + offsets, cur_inp, (~fill_mask) & mask)
    tl.store(out + offsets, value, fill_mask & mask)

def masked_fill(inp, mask, value):
    # ... 参数验证代码 ...
    # inp.device = "npu"
    out = torch.zeros_like(inp)
    N = inp.numel()
    if N == 0:
        return out

    grid = lambda meta: (triton.cdiv(N, 4096),)  # 导致 coreDim 超限
    masked_fill_kernel[grid](inp, mask.to(torch.int), value, out, N, 4096)
    return out
```


Optimized code:


```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
    BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    # 计算需要处理的子块数量
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    # 分块处理，避免 UB 溢出
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        # 分批加载和处理数据
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        # 先写入原始数据
        tl.store(out + offsets, input_vals, mask=mask)
        # 然后在需要填充的位置覆写目标值
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)

def masked_fill(inp, expand_mask, value):
    logger.debug("GEMS MASKED FILL")

    out = torch.zeros_like(inp)
    # ... 参数验证代码 ...
    # inp.device = "npu"
    N = inp.numel()
    if N == 0:
        return out

    # 使用优化的参数配置
    MAIN_BLOCK_SIZE = 32768  # 确保 coreDim 合规
    SUB_BLOCK_SIZE = 1024    # 控制 UB 使用量

    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, N,
                        MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
```


### Why does the UBSIZE out-of-memory error occur


The segmentation is unreasonable, with excessive unaligned memory access or computation. For example, when moving 2D data of size (64, 32), the corresponding stride is (12832, 128). For aligned data access, the corresponding stride should be (32, 1). For unaligned access content, add a new axis of size 1 to the innermost dimension, changing it to (64, 32, 4). Since the hardware requires UB memory to be 32-byte aligned in vector operator scenarios, assuming `type=float16`, the corresponding stride should be (12832, 128, 1).


### Line-by-Line Comparison of Discrete Memory Access Code to Observe Inefficient Scalar Mapping


Set the environment variable `TRITON_DEBUG=1`, save `~/.triton/cache/xxx.ttadapter`, and then execute.


```diff
bishengir-compile xxx.ttadapter --target=Ascend910B3 --enable-auto-multi-buffer=True --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync
```


The IR will be output. Compare the Triton operator logic with the operations inside the IR to observe whether there are any operations that have not been mapped to instructions. Observe whether there is pure scalar data movement or computation in the HIVM IR stage that has not been mapped to SIMD instructions, as this could become a performance bottleneck.


Problem: Discontinuous memory access && inefficient scalar mapping  
b[1024, 32] = a[1024, 32] The original Triton implementation uses a thread-based approach, binding the lowest dimension 32 in [1024, 32] to thread blocks, then splitting 1024 by 16 into [64, 16, 32], and finally binding 64 to thread blocks.


```diff
chunk_fwd_kernel_o[(NT, B * H)](
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(1024,), # 一维张量
        strides=(32,), # 连续内存
        offsets=(i_t * 16,), # 从起始位置开始
        block_shape=(BT,), # 块大小
        order=(0,) # 连续访问
    )
​)
```


Optimization Approach


Adjust the shape/stride of `block_ptr`:

Treat (1024, 32) as a two-dimensional matrix, where the lowest dimension 32 is contiguous, so the stride should be (32, 1) instead of (32,). This allows each thread block to access 32 contiguous elements. Bind the thread block to the row dimension (1024), with each thread processing an entire row of 32 elements. This ensures contiguous memory access and good locality.


For example:


```diff
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024, 32),
    strides=(32, 1),
    offsets=(i_t * BT, 0),
    block_shape=(BT, 32),
    order=(1, 0) # 行优先布局：维度 1 最连续（stride 1），维度 0 最不连续
)
```

