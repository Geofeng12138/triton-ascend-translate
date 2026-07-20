# triton.language.broadcast


## 1 Function Description


Broadcast two tensors to a common compatible shape so that they can perform element-wise operations.


**Syntax:**


- `triton.language.broadcast(input, other)` - function call form


**Function:**


- Automatically align tensors of different ranks to obtain the target shape  
- Expand dimensions of size 1 to match the corresponding dimension size in the target shape


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | The first input tensor, must be of type RankedTensorType |
| other | tensor | Yes | The second input tensor, must be of type RankedTensorType |


**Return value:**


- **Type:** tensor  
- **Shape:** The target shape that two tensors are compatible with together  
- **Data Type:** Each returned tensor retains the original data type of its input  
- **Memory Layout:** Returns a newly created tensor


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


**Basic Usage:**


```python
@triton.jit
def broadcast_kernel(
    output_ptr,
    BLOCK_SIZE: tl.constexpr
):
    # 创建一个标量（0维张量）
    scalar = 5.0

    # 创建一个向量（1维张量）
    vector = tl.arange(0, BLOCK_SIZE) * 1.0  # 形状: (BLOCK_SIZE,)

    # 使用 broadcast 将标量广播到与向量相同的形状
    # scalar: () -> (BLOCK_SIZE,)
    broadcasted_scalar = tl.broadcast(scalar, vector)

    result = vector + broadcasted_scalar

    # 存储结果
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + offsets, result)

```

