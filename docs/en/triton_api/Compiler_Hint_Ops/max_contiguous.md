# triton.language.max_contiguous


## 1. Function Overview


`max_contiguous` is used to declare the contiguity pattern in the input tensor to the compiler, indicating that the first `value` dimensions of the input tensor are contiguous.


```python
triton.language.max_contiguous(input, values, _builder=None, _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `input` | `Tensor` | Required | Input tensor whose memory access has a specific contiguous pattern |
| `values` | `constexpr[int]` or `list[constexpr[int]]` | Required | Compile-time constant integer (or sequence of integers) describing the contiguous pattern |
| `_semantic` | - | - | Reserved parameter, external calls not supported for now |


**`values` describes the continuous features of each dimension, so the dimensions of `values` must match those of `input`.  
Note the dimensionality reduction that occurs when the last dimension of `shape` is `1`.**


For example: the two-dimensional `input` corresponds to the general `values` input parameter as `[1,1]`.


### 2.2 Type Support


| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | × | × | ✓ | ✓ | ✓ | × | ✓ | ✓ |


### 2.3 Special Restrictions


Relative community capability deficiency and inability to achieve


Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).


### 2.4 Usage


```python
@triton.jit
def triton_max_contiguous(A, B, BLOCK_SIZE : tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    val = tl.load(A + offsets)
    # 声明offset里的前BLOCK_SIZE个数是连续的
    input_data = tl.max_contiguous(val, [BLOCK_SIZE])

    # 编译器可以生成更高效的内存访问指令
    result = input_data* 2
    tl.store(B + offsets, result)
```

