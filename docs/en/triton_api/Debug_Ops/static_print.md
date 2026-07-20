# triton.language.static_print


## 1. Function Overview


`static_print` is used to print information at compile time, similar to Python's `print()` function, but it executes during kernel compilation rather than at runtime.


```python
triton.language.static_print(*values, sep: str = ' ', end: str = '\n', file=None, flush=False, _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `tensor`/`scalar` | Required | Values to print, supports multiple parameters |
| `sep` | `str` | `' '` | Separator between values |
| `end` | `str` | `'\n'` | Suffix at the end of printing |
| `file` | - | - | File object to write to |
| `flush` | `bool` | `False` | Whether to flush the output buffer |
| `_semantic` | - | - | Reserved parameter, external calls not supported for now |


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


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.


### 2.3 Special Limitations


Relative community capability deficiency and inability to achieve


Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).


### 2.4 Usage


```python
import triton.language as tl

@triton.jit
def basic_static_print_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # 在编译时打印常量的值
    tl.static_print("BLOCK_SIZE =", BLOCK_SIZE)
    tl.static_print(BLOCK_SIZE)
    # 支持fstring打印方式
    tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
```


If printing a **non-constant** result, it will print a value in the format `data type[data shape (scalar is empty)]`. For example, if the data type pointed to by `x_ptr` in the code below is `int32`, it will print the result `val:int32[constexpr[4]]`.


```python
import triton.language as tl

@triton.jit
def basic_static_print_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, 4)
    val = tl.load(x_ptr + idx)
    tl.static_print("val:",val)
    #非常量不支持fstring打印
    #tl.static_print(f"val:{val}")
```

