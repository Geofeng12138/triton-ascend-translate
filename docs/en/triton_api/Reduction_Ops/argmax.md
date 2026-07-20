# triton.language.argmax


## 1. OP Overview


Introduction: Returns the index of the maximum value along the specified dimension.


```python
triton.language.argmax(input, axis, tie_break_left=True, keep_dims=False)
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter | Type | Description |
| :---: | :---: | --- |
| `input` | `tensor` | Tensor data, left operand |
| `axis` | `int` | Specifies which dimension to reduce over |
| `keep_dims` | `bool` | Whether to keep the reduced dimensions |
| `tie_break_left` | `bool` | If multiple elements have the same maximum value, return the index of the leftmost maximum |


Return value:  
`tl.tensor`: A tensor with the same shape as `input`.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2A3 series| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| -------- | ---------------------- |
| GPU    | No limit |
| Ascend | No limit |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms.


### 2.3 Special Limitations


Relative community capability deficiency and inability to achieve


Ascend A3 lacks support for uint16, uint32, uint64, and fp64 compared to GPU.


### 2.4 Usage


For more examples, refer to the triton-ascend code repository, ascend/examples/generalization_cases/test_argmax.py


```@triton.jit
def triton_argmax_1d(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.argmax(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)
```


## 3. Special Value Cases


For the case of tensor[nan, inf], return the index where inf is located.

