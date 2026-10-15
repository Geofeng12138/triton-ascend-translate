# div


## 1. OP Overview


Introduction: Division, the arithmetic operation '/', without the `tl.div` method.


The underlying implementation is the same as the `fdiv` operator, except that `fdiv` explicitly restricts the input parameters to be of float type, while `/` has no such restriction — it converts non-float types to float before performing the calculation.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type                | Description                                                  |
| ------------- | ------------------- | ------------------------------------------------------------ |
| `self`        | `tensor or Number`  | First input parameter, the dividend                          |
| `other`       | `tensor or Number`  | Second input parameter, the divisor                          |


Return value:  
`tl.tensor`: The result of division  
Return type: Always returns a floating-point type


| Input Type          | Handling Method               | Result Type    |
| ------------------- | ----------------------------- | --------------- |
| `int / int`         | Both converted to `float32`   | `float32`       |
| `int / float`       | int converted to float        | float type      |
| `float / float`     | Unified to higher precision float | Higher precision float |
| `float / int`       | int converted to float        | float type      |


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| × | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Unlimited               |
| Ascend A2/A3 | Unlimited               |


Conclusion: There is no difference between GPU and Ascend platform in terms of Shape.


### 2.3 Special Restrictions


Ascend A3 lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU.


### 2.4 Usage


The following example implements division calculation on the input tensors `in_ptr0, in_ptr1`:


```python
@triton.jit
def triton_div(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 / tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```

