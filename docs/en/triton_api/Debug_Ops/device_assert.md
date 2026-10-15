# triton.language.device_assert


**To use `device_assert`, the environment variable `TRITON_DEBUG` must be set to a non-zero value to take effect.**


## 1. Function Overview


`device_assert` is used to perform assertion checks from the device side during GPU runtime, outputting an error message if the condition is not met.


```python
triton.language.device_assert(cond, msg='', _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cond` | `bool` | Required | The condition expression to assert at runtime |
| `msg` | `str` | `''` | Error message displayed when assertion fails |
| `_semantic` | - | - | Reserved parameter, external calls not supported for now |


### 2.2 Type Support


A3:


| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |


### 2.3 Usage


```python
import triton.language as tl

@triton.jit
def basic_device_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # 基本断言：检查程序ID
    pid = tl.program_id(0)
    tl.device_assert(pid >= 0, "Program ID must be non-negative")

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)

    # 检查数据有效性（比如检查张量中没有负值）
    tl.device_assert(tl.min(x) >= 0, "All values must be non-negative")
```

