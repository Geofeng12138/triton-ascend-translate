# triton.language.static_assert


## 1. Function Overview


`static_assert` is used to assert whether a condition holds at compile time. If the condition is not satisfied, compilation fails. This is a compile-time checking tool that does not require setting debug environment variables.


```python
triton.language.static_assert(cond, msg='', _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cond` | `bool` | Required | The condition expression to assert at compile time |
| `msg` | `str` | `''` | Error message displayed when assertion fails |
| `_semantic` | - | - | Reserved parameter, currently not supported for external calls |


### 2.2 Type Support


A3:


| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |


**Note:** The type of the value in the `cond` statement must be `constexpr`.


### 2.3 Usage


```python
import triton.language as tl

@triton.jit
def basic_static_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # 基本断言：检查BLOCK_SIZE是否为2的幂次
    tl.static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)

    # 带自定义错误消息的断言
    tl.static_assert(BLOCK_SIZE >= 64, "BLOCK_SIZE must be at least 64 for performance")

    # 在static_assert的条件中出现非常量会编译错误
    # val = tl.load(x_ptr)
    # tl.static_assert(val <= 64)
```

