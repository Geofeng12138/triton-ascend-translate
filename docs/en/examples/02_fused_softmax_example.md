# Fused Softmax


In this section, we will write a fused softmax operation program using Triton. Along the way, you will learn:


- The advantages of kernel fusion for bandwidth-limited operations.
- Reduction operations in Triton.


## Using native PyTorch to perform row-wise Softmax computation on X


```Python
import torch
import torch_npu

import triton
import triton.language as tl

def naive_softmax(x):
    """
    我们减去最大元素以避免溢出。Softmax 对于这种偏移是不变的。
    """
    # 读取 MN 个元素；写入 M 个元素
    x_max = x.max(dim=1)[0]
    # 读取 MN + M 个元素；写入 MN 个元素
    z = x - x_max[:, None]
    # 读取 MN 个元素；写入 MN 个元素
    numerator = torch.exp(z)
    # 读取 MN 个元素；写入 M 个元素
    denominator = numerator.sum(dim=1)
    # 读取 MN + M 个元素；写入 MN 个元素
    ret = numerator / denominator[:, None]
    # 总计：读取 5MN + 2M 个元素；写入 3MN + 2M 个元素
    return ret
```


The purpose of kernel fusion


When implemented natively in PyTorch, computing `y=naive_softmax(x)` requires reading 5MN+2M elements from DRAM and writing back 3MN+2M elements. This is clearly very inefficient; we would prefer to use a custom "fused" kernel that reads x only once and performs all necessary computations on-chip.

This way, only 2MN bytes need to be read and written back, so we can expect a theoretical speedup of approximately 4x (i.e., (8MN+4M)/2MN).


`torch.jit.script` aims to automatically perform this kind of "kernel fusion," but it is still far from ideal.


## Compute Kernel


The softmax kernel works as follows: each compute unit (program) loads a set of rows of the input matrix X in strides equal to the number of programs, performs normalization, and writes the result to the output matrix Y.

Note: An important limitation of Triton is that each block must have a power-of-two number of elements. Therefore, if we need to handle arbitrary input shapes, we must internally "pad" each row and ensure memory operation correctness.


```Python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # 程序起始行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # 步长表示我们需要对指针增加多少以推进 1 行
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # 块大小是大于 n_cols 的下一个二的幂，因此我们可以适配
        # 单个块中的行
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # 将行加载到 SRAM 中，使用掩码，因为 BLOCK_SIZE 可能大于 n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # 为了数值稳定性而减去最大值
        row_minus_max = row - tl.max(row, axis=0)
        # 请注意，Triton 中的指数运算速度很快，但是是近似的。
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # 将输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```


We can create a helper function that adds the kernel function and its meta-parameters to the execution queue to process any given input tensor.


```Python
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape

    # 每次循环迭代的块大小是大于或等于`x`列数的最小二的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 分配输出空间
    y = torch.empty_like(x)

    # 预编译内核以获取寄存器使用情况并计算线程占用情况。
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        num_programs = 32
        kernel = softmax_kernel
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE
    )
    return y
```


## Unit Tests


Need to test the processed kernel on a matrix with irregular numbers of rows and columns, which can verify whether the padding mechanism is working.


```Python
torch.manual_seed(0)
x = torch.randn(1823, 781, device='npu')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
print(y_triton)
print(y_torch)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(y_triton-y_torch))}')
```


Out:


```bash
tensor([[0.0002, 0.0017, 0.0009,  ..., 0.0009, 0.0013, 0.0073],
        [0.0001, 0.0004, 0.0006,  ..., 0.0006, 0.0004, 0.0003],
        [0.0007, 0.0002, 0.0006,  ..., 0.0011, 0.0004, 0.0039],
        ...,
        [0.0021, 0.0002, 0.0015,  ..., 0.0012, 0.0014, 0.0022],
        [0.0003, 0.0002, 0.0007,  ..., 0.0005, 0.0006, 0.0007],
        [0.0034, 0.0014, 0.0005,  ..., 0.0007, 0.0016, 0.0028]],
       device='npu:0')
tensor([[0.0002, 0.0017, 0.0009,  ..., 0.0009, 0.0013, 0.0073],
        [0.0001, 0.0004, 0.0006,  ..., 0.0006, 0.0004, 0.0003],
        [0.0007, 0.0002, 0.0006,  ..., 0.0011, 0.0004, 0.0039],
        ...,
        [0.0021, 0.0002, 0.0015,  ..., 0.0012, 0.0014, 0.0022],
        [0.0003, 0.0002, 0.0007,  ..., 0.0005, 0.0006, 0.0007],
        [0.0034, 0.0014, 0.0005,  ..., 0.0007, 0.0016, 0.0028]],
       device='npu:0')
The maximum difference between torch and triton is 1.4901161193847656e-08
```


"The maximum difference between torch and triton is 1.4901161193847656e-08" indicates that the output results of Triton and PyTorch are very close and indistinguishable to the naked eye.

