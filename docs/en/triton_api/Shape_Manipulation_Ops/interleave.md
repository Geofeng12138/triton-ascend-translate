# triton.language.interleave


## 1 Function Description


Interleave two input tensors of the same shape along the last dimension. The size of the last dimension of the output tensor is twice that of the input tensors, while the other dimensions remain unchanged.


**Syntax:**


- `triton.language.interleave(x, y)` - function call form
- `x.interleave(y)` - member function form


**Function:**


- Interleave two input tensors of the same shape along the last dimension
- The last dimension of the output tensor is twice the size of the input tensors
- Other dimensions remain unchanged


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|--------|------|------|------|
| x | tensor | Yes | The first input tensor |
| y | tensor | Yes | The second input tensor, shape must be the same as x |


**Return value:**


- **Type:** tensor
- **Shape:** The last dimension of the input shape multiplied by 2
- **Data type:** Same as the input tensor
- **Memory layout:** Interleaved arrangement of x and y elements


**Constraints:**


- The two input tensors must have the same shape and data type  
- The shape of the output tensor is the last dimension of the input shape multiplied by 2


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
import triton
import triton.language as tl

@triton.jit
def interleave_example():
    # 创建两个2x3的张量
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.ones([2, 3], dtype=tl.float32)

    # 交错排列，变成2x6
    z = tl.interleave(x, y)

    return z

## 调用示例
result = interleave_example()
print(result.shape)  # 输出: (2, 6)
```

