# triton.language.view


## 1 Function Description


Create a view of a tensor, changing its shape without copying data, similar to `reshape`, but with a stronger emphasis on the concept of a view, maintaining the continuity of data in memory.


**Syntax:**


- `triton.language.view(input, shape)` - function call form
- `input.view(shape)` - member function form


**Function:**


- Create a view of a tensor, changing its shape without copying the data  
- Similar to `reshape`, but with a stronger emphasis on the concept of a view  
- Maintains the contiguity of data in memory


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|----------------|------|----------|-------------|
| input | tensor | Yes | Input tensor |
| shape | List[int] | Yes | Target shape |


**Return value:**


- **Type:** tensor
- **Shape:** Same as the target shape specified by the `shape` parameter
- **Data type:** Same as the input tensor
- **Memory layout:** Contiguous in memory with the input tensor


**Constraints:**


- The total number of elements in the input and output tensors must be equal
- The output tensor must be contiguous in memory with the input tensor


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
def view_example(out_ptr):
    # 创建2x3x4的张量
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # 创建视图，变成6x4
    y = tl.view(x, [6, 4])

    # 将结果写回外部张量
    offs = tl.arange(0, 6)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offs, y)

## 调用示例
out = torch.empty((6, 4), dtype=torch.float32, device="npu")
view_example[(1,)](out)
print(out.shape)  # 输出: torch.Size([6, 4])
```

