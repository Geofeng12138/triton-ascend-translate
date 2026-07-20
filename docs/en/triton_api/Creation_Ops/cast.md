# triton.language.cast


## 1 Function Description


Convert the tensor to the specified data type, supporting numeric type conversion, bit-level reinterpretation (bitcast), floating-point precision reduction rounding mode, and Ascend-extended integer overflow handling mode.


**Syntax:**


- `triton.language.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)` - Function call form
- `input.cast(dtype, fp_downcast_rounding=None, bitcast=False)` - Member function form


**Function:**


- Numeric type conversion: integer <-> integer, float <-> float, integer <-> float
- Bit-level reinterpretation (bitcast): bits unchanged, only the interpretation type changes
- Float precision reduction supports rounding modes: `rtne` (default, round to nearest, ties to even), `rtz` (toward zero)
- Integer conversion (Ascend extension) supports overflow modes: `trunc` (truncation, default), `saturate` (saturation)


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|--------|------|------|------|
| input | tensor | Yes | Input tensor |
| dtype | tl.dtype | Yes | Target data type |
| fp_downcast_rounding | str | No | Only valid for floating-point downcasting, `rtne` or `rtz` |
| bitcast | bool | No | Whether to perform bit-level reinterpretation, default False |
| overflow_mode | str | No | Ascend extension: integer overflow handling, `trunc` or `saturate` |


**Return value:**


- **Type:** tensor
- **Shape:** Same as the input tensor
- **Data type:** Same as the target type specified by the `dtype` parameter
- **Memory layout:** Whether to perform bit-level reinterpretation is determined by the `bitcast` parameter


**Constraints:**


- `fp_downcast_rounding` can only be set when downcasting floating-point precision; otherwise, an error will be reported.
- When `bitcast=True`, no numerical conversion is performed, and rounding/overflow modes are ignored.
- `overflow_mode` is only meaningful for integer types (Ascend extension).


### 2.2 DataType Support Table


| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


### 2.3 Shape Support Table


Supports any number of dimensions and any shape size.


### 2.4 Special Limitations


None


### 2.5 Usage


**Basic Usage:**


```python
import triton
import triton.language as tl

@triton.jit
def cast_example():
    # 创建float32张量
    x = tl.zeros([2, 3], dtype=tl.float32)

    # 转换为int32
    y = tl.cast(x, tl.int32)

    return y

## 调用示例
result = cast_example()
print(result.dtype)  # 输出: int32
```


**Advanced Usage:**


```python
@triton.jit
def cast_advanced_example():
    # 创建float32张量
    x = tl.zeros([2, 3], dtype=tl.float32)

    # 位级别重解释
    y = x.cast(tl.int32, bitcast=True)

    # 浮点降精度，向零舍入
    z = x.cast(tl.float16, fp_downcast_rounding="rtz")

    # float32 → int8，启用饱和模式（Ascend 扩展，超出 int8 范围的值会被截断到 [-128, 127]）
    w = x.cast(tl.int8, overflow_mode="saturate")

    return y, z, w
```


**Practical Application Scenarios:**


```python
@triton.jit
def quantization_kernel(x_ptr, output_ptr, scale, zero_point, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 加载float32数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 量化：转换为int8
    x_quantized = tl.cast(x * scale + zero_point, tl.int8, overflow_mode="saturate")

    # 存储量化结果
    tl.store(output_ptr + offsets, x_quantized, mask=mask)
```

