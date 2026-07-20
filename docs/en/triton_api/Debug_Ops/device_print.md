# triton.language.device_print


## 1. Function Overview


`device_print` is used to print information from the device side during NPU runtime. Unlike `static_print`, it outputs information in real-time while the kernel is executing. The first parameter must be a `string`, and subsequent parameters must be `scalars` or `tensors`. **To use `device_print`, the environment variable `TRITON_DEVICE_PRINT` must be set to `True`.**


```python
triton.language.device_print(prefix, *args, hex=False, _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str` | Required | Prefix string before the printed value |
| `args` | `tensor`/`scalar` | Required | Value(s) to print, can be any tensor or scalar |
| `hex` | `bool` | `False` | Whether to print all values in hexadecimal format |
| `_semantic` | - | - | Reserved parameter, external calls not supported yet |


### 2.2.1 Data Type Support


A3:


| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | × | × | ✓ | ✓ | ✓ | × | ✓ | ✓ |


### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensor |
| Ascend | Only supports 1~5D tensor |


### 2.3 Special Limitations


Relative community capability is lacking and cannot be achieved.


Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).


### 2.4 Usage


**Note**: The `prefix` string prefix must be added when using `device_print`, otherwise it will cause a compilation error.


```python
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr):
    idx = tl.arange(0,3)
    idy = tl.arange(0,4)
    offset = idx[:,None] * 4 + idy[None,:]
    val = tl.load(x_ptr + offset)
    # 打印二维张量val的值
    tl.device_print("val:",val)
```

