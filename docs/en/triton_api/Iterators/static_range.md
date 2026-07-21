# triton.language.static_range


## 1. Function Overview


`static_range` is a static range iterator, similar to `range` but performs aggressive loop unrolling optimization at compile time.


```python
triton.language.static_range(arg1, arg2=None, step=None, _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `arg1` | `constexpr` | Required | Start value (when used as a single parameter, it serves as the end value, starting from 0) |
| `arg2` | `constexpr` | - | End value (not included in the range) |
| `step` | `constexpr` | `1` | Step increment for each iteration |
| `_semantic` | - | - | Reserved parameter, external calls not currently supported |


### 2.2 Type Support


A3:


| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend 910 Series | ✓ | ✓ | ✓ | × | × | × | × | ✓ | × | × | × | × | × |


### 2.3 Special Restrictions Explanation


Relative community capability is lacking and cannot be achieved.


Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).


### 2.4 Usage


```python
@triton.jit
def optimized_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # 使用static_range进行小规模循环展开，消除循环开销
    for i in tl.static_range(BLOCK_SIZE):
        # 当BLOCK_SIZE是编译时常量时，整个循环会被展开
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)

    # 对比：使用range会有循环控制开销
    for i in tl.range(BLOCK_SIZE):
        # 这个循环在运行时会有循环控制逻辑
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)
```


`static_range` trades off code size for runtime performance, making it suitable for scenarios with known and small loop iteration counts.

