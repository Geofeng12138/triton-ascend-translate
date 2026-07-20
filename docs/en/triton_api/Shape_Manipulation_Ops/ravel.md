# triton.language.ravel


## 1 Function Description


Flatten the input tensor into a one-dimensional tensor, preserving the order of elements in memory. The total number of elements in the output tensor is the same as that of the input tensor.


**Syntax:**


- `triton.language.ravel(input)` - function call form
- `input.ravel()` - member function form


**Features:**


- Flatten the input tensor into a one-dimensional tensor
- Preserve the order of elements in memory
- The total number of elements in the output tensor is the same as that in the input tensor


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|----------------|------|----------|-------------|
| input | tensor | Yes | Input tensor |


**Return value:**


- **Type:** tensor
- **Shape:** 1D tensor containing all elements of the input tensor
- **Data type:** Same as the input tensor
- **Memory layout:** Flattened in row-major order


**Constraints:**


- No special constraints, supports input of any shape.


### 2.2 DataType Support Table


| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


### 2.3 Shape Support Table


Supports any number of dimensions and any shape size.


### 2.4 Special Restrictions


No


### 2.5 Usage


```python
@triton.jit
def flatten_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # 加载2D数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 展平为一维
    x_flat = x.ravel()

    # 存储展平结果
    tl.store(output_ptr + offsets, x_flat, mask=mask)
```

