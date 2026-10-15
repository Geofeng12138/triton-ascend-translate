# triton.language.broadcast_to


## 1 Function Description


Broadcast the tensor to the target shape, automatically handling dimension alignment. The broadcast operation does not copy data; instead, it is achieved by changing the tensor's shape and stride.


**Syntax:**


- `triton.language.broadcast_to(input, shape)` - function call form
- `input.broadcast_to(shape)` - member function form


**Function:**


- Automatically handle dimension alignment by expanding dimensions of size 1 to match the corresponding dimension size in the target shape.  
- Keep the data unchanged, only modify the shape information of the tensor.


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |
| shape | List[int] | Yes | Target shape |


**Return value:**


- **Type:** tensor
- **Shape:** Same as the target shape specified by the `shape` parameter
- **Data type:** Same as the input tensor
- **Memory layout:** Broadcasting is achieved by modifying stride information, with no data copy


**Constraints:**


- The number of dimensions of the input tensor must equal the number of dimensions of the target shape.
- All dimensions must satisfy the broadcasting rules.


### 2.2 DataType Support Table


| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


### 2.3 Shape Support Table


Supports any number of dimensions and any shape size.


### 2.4 Special Limitations


Unlike broadcast, the broadcast_to implemented by the Triton community must ensure that the rank of the tensor's shape matches the rank of the target shape.


### 2.5 Usage


**Basic Usage:**


```python
@triton.jit
def matrix_add_bias_kernel(x_ptr, bias_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 加载数据块
    x = tl.load(x_ptr + offsets, mask=mask)

    # 广播bias到匹配的形状
    bias = tl.load(bias_ptr)
    bias_broadcast = bias.broadcast_to([BLOCK_M, BLOCK_N])

    # 执行加法
    output = x + bias_broadcast
    tl.store(output_ptr + offsets, output, mask=mask)
```

