# triton.language.expand_dims


## 1 Function Description


Insert a dimension of size 1 at the specified axis position without changing the tensor's data, only increasing the number of dimensions. Supports negative indexing, counting from right to left.


**Syntax:**


- `triton.language.expand_dims(input, axis)` - function call form
- `input.expand_dims(axis)` - member function form


**Function:**


- Insert a dimension of size 1 at the specified axis position  
- Does not change the tensor data, only increases the number of dimensions  
- Supports negative indexing, counting from right to left


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | tensor | Yes | Input tensor |
| axis | int \| Tuple[int] | Yes | Position to insert the dimension, supports negative indexing |


**Return value:**


- **Type:** tensor  
- **Shape:** Inserts a dimension of size 1 at the specified axis position  
- **Data Type:** Same as the input tensor  
- **Memory Layout:** Implemented via `tensor::ExpandShapeOp`, no data copy


**Constraints:**


- axis must be within the range [-rank-1, rank], where rank is the number of dimensions of the input tensor
- The size of the inserted dimension is fixed at 1


### 2.2 DataType Support Table


| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


### 2.3 Shape Support Table


Supports any number of dimensions and any shape size.


### 2.4 Special Limitations


None


### 2.5 Usage


```python
import torch
import triton
import triton.language as tl

@triton.jit
def expand_dims_example(out_ptr):
    # 创建2x3的张量
    x = tl.zeros([2, 3], dtype=tl.float32)

    # 在axis=1位置插入维度，变成2x1x3
    y = tl.expand_dims(x, axis=1)

    # 将结果写回外部张量
    offs = (
        tl.arange(0, 2)[:, None, None] * 3
        + tl.arange(0, 1)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, y)

## 调用示例
out = torch.empty((2, 1, 3), dtype=torch.float32, device="npu")
expand_dims_example[(1,)](out)
print(out.shape)  # 输出: torch.Size([2, 1, 3])
```

