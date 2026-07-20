# triton.language.add


## 1. OP Overview


Introduction: Addition, equivalent to the arithmetic operator '+'.  
Prototype:


```python
triton.language.add(x, y, sanitize_overflow: constexpr = True, _builder=None)
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x` | `tensor or Number` | The first input parameter |
| `y` | `tensor or Number` | The second input parameter |
| `sanitize_overflow` | `bool` | Whether to perform overflow checking for integer addition. The default value is True and does not need to be explicitly specified. |
| `_builder` | - | Reserved parameter, currently not supported for external calls |


Return value:  
`tl.tensor`: The result of addition


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Unlimited               |
| Ascend | Unlimited               |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms.


### 2.3 Usage


The following example implements addition on the input tensors `x_ptr, y_ptr`:


```python
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y # 等价于 output = tl.add(x,y)
    tl.store(output_ptr + offsets, output, mask=mask)
```


## 3. Special Notes


Ascend A3 vs GPU: fp64 not supported

