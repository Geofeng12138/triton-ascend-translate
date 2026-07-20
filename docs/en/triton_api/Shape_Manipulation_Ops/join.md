# triton.language.join


## 1 Function Description


Concatenates two input tensors of the same shape along a new smallest dimension. The output tensor has one more dimension than the input tensors, with a size of 2, while keeping the other dimensions unchanged.


**Syntax:**


- `triton.language.join(x, y)` - function call form
- `x.join(y)` - member function form


**Features:**


- Concatenate two input tensors of the same shape along a new minimum dimension
- The output tensor has one more dimension than the input tensors, with a size of 2
- Keep other dimensions unchanged


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|----------------|------|----------|-------------|
| x | tensor | Yes | The first input tensor |
| y | tensor | Yes | The second input tensor |


**Return value:**


- **Type:** tensor
- **Shape:** The shape of the input tensor after broadcasting, plus a dimension of size 2
- **Data type:** Same as the input tensor
- **Memory layout:** Stack x and y along the new dimension


**Constraints:**


- The two input tensors must have shapes and data types that can be broadcast to the same shape.


### 2.2 DataType Support Table


| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


### 2.3 Shape Support Table


Supports any number of dimensions and any shape size.


### 2.4 Special Restrictions Explanation


None


### 2.5 Usage


```python
import torch
import triton
import triton.language as tl

@triton.jit
def join_example(out_ptr):
    # 创建两个2x3的张量
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.full([2, 3], 1.0, dtype=tl.float32)

    # 连接，变成2x2x3
    z = tl.join(x, y)

    # 将结果写回外部张量
    offs = (
        tl.arange(0, 2)[:, None, None] * (2 * 3)
        + tl.arange(0, 2)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, z)

## 调用示例
out = torch.empty((2, 2, 3), dtype=torch.float32, device="npu")
join_example[(1,)](out)
print(out.shape)  # 输出: torch.Size([2, 2, 3])
```

