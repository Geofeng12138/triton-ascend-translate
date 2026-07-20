# triton.language.max_constancy


## 1. Function Overview


`max_constancy` is used to declare to the compiler the constancy pattern of values in the input tensor, informing the compiler that each group of consecutive values in the input data is equal.


```python
triton.language.max_constancy(input, values, _builder=None, _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `Tensor` | Required | Input tensor whose values have a specific constancy pattern |
| `values` | `constexpr[int]` or `list[constexpr[int]]` | Required | Compile-time constant integer (or sequence of integers) describing the constancy pattern |
| `_semantic` | - | - | Reserved parameter, currently not supported for external calls |


**`values` describes the constancy characteristics of each dimension, so the dimensions of `values` must match those of `input`.  
Note the dimensionality reduction that occurs when the last dimension of `shape` is `1`.**


For example: the two-dimensional `input` corresponds to the general `values` input parameter as `[1,1]`.


### 2.2 Type Support


A3:


| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | × | × | ✓ | ✓ | ✓ | × | ✓ | ✓ |


### 2.3 Special Restrictions Explanation


Relative community capacity deficiency and inability to achieve


Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).


### 2.4 Usage


```python
@triton.jit
def basic_constancy_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)

    # 使用constexpr声明常量性模式：每4个连续的值都是相等的
    # 例如输入模式：[0,0,0,0,1,1,1,1,2,2,2,2,...]
    input_data = tl.max_constancy(input_data, [4])

    # 编译器可以基于这个信息进行优化
    result = input_data * 2
    tl.store(B + offsets, result)
```

