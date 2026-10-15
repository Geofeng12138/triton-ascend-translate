# triton.language.permute


## 1 Function Description


Rearranges the dimensions of a tensor according to the `dims` parameter, without changing the tensor's data, only the order of the dimensions. Supports rearrangement of any number of dimensions.


**Syntax:**


- `triton.language.permute(input, dims)` - function call form
- `input.permute(dims)` - member function form


**Function:**


- Rearranges the dimensions of a tensor according to the `dims` parameter
- Does not change the tensor data, only the order of dimensions
- Supports arbitrary dimension rearrangement


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |
| dims | List[int] | Yes | New dimension order |


**Return value:**


- **Type:** tensor
- **Shape:** dimensions rearranged according to the `dims` parameter
- **Data type:** same as the input tensor
- **Memory layout:** transposition achieved by modifying stride information, no data copy


**Constraints:**


- dims must contain all dimension indices of the input tensor


### 2.2 DataType Support Table


| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


### 2.3 Shape Support Table


Supports any number of dimensions and any shape size.


### 2.4 Special Limitations


* Transpose for dimensions higher than 8 is not supported.


### 2.5 Usage


```python
import torch
import triton
import triton.language as tl

@triton.jit
def permute_example(out_ptr):
    # 创建2x3x4的张量
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # 转置维度，变成4x2x3
    y = tl.permute(x, [2, 0, 1])

    # 将结果写回外部张量
    offs = (
        tl.arange(0, 4)[:, None, None] * (2 * 3)
        + tl.arange(0, 2)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, y)

## 调用示例
out = torch.empty((4, 2, 3), dtype=torch.float32, device="npu")
permute_example[(1,)](out)
print(out.shape)  # 输出: torch.Size([4, 2, 3])
```

