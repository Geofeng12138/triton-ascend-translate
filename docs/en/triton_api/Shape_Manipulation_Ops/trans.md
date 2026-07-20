# triton.language.trans


## 1 Function Description


Transposes the dimensions of a tensor according to the `dims` parameter, without changing the tensor's data, only the order of the dimensions. A specially optimized transpose operation.


**Syntax:**


- `triton.language.trans(input, dims)` - Function call form
- `input.trans(dims)` - Member function form


**Features:**


- Transpose the dimensions of a tensor according to the `dims` parameter  
- Does not change the tensor data, only the order of dimensions  
- Specially optimized transpose operation


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|----------------|------|----------|-------------|
| input | tensor | Yes | Input tensor |
| dims | List[int] | Yes | Dimension order after transposition |


**Return value:**


- **Type:** tensor
- **Shape:** dimensions rearranged according to the `dims` parameter
- **Data type:** same as the input tensor
- **Memory layout:** transpose achieved by modifying stride information, no data copy


**Constraints:**


- dims must include all dimension indices of the input tensor


### 2.2 DataType Support Table


| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


### 2.3 Shape Support Table


Supports any number of dimensions and any shape size.


### 2.4 Special Restrictions


* Transposition of dimensions higher than 8 is not supported


### 2.5 Usage


```python
import triton
import triton.language as tl

@triton.jit
def trans_example():
    # 创建2x3x4的张量
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # 转置维度，变成4x2x3
    y = tl.trans(x, [2, 0, 1])

    return y

## 调用示例
result = trans_example()
print(result.shape)  # 输出: (4, 2, 3)
```

