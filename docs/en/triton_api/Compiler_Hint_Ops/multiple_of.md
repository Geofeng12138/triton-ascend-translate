# triton.language.multiple_of


## 1. Function Overview


`multiple_of` is used to declare to the compiler that the first value in the input tensor is a multiple of a certain number.


```python
triton.language.multiple_of(input, values, _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `Tensor` | Required | Input tensor, whose values are all multiples of a certain number |
| `values` | `constexpr[int]` or `list[constexpr[int]]` | Required | Declares that the input values are multiples of these numbers (a single integer or a sequence of integers per dimension) |
| `_semantic` | - | - | Reserved parameter, currently not supported for external calls |


**`values` describes the divisibility characteristics of the first value in each dimension, so the dimensions of `values` must match those of `input`.
Note the dimensionality reduction that occurs when the last dimension of `shape` is `1`.**


For example: the two-dimensional `input` corresponds to the general `values` input parameter as `[1,1]`.


### 2.2 Type Support


A3:


| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |


### 2.3 Special Restrictions Explanation


Relative community capability is lacking and cannot be achieved.


Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).


### 2.4 Usage


```python
@triton.jit
def basic_multiple_of_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)

    # 声明输入张量的第一个值是BLOCK_SIZE的倍数
    input_data = tl.multiple_of(input_data, [BLOCK_SIZE])
```

