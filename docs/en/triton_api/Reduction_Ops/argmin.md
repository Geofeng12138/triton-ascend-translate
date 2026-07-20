# triton.language.argmin


## 1. OP Overview


Introduction: Returns the index of the minimum value along the specified dimension.


```python
triton.language.argmin(input, axis, tie_break_left=True, keep_dims=False)
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter | Type | Description |
| :---: | :---: | --- |
| `input` | `tensor` | Tensor data, left operand |
| `axis`   | `int` | Specifies the dimension along which to reduce |
| `keep_dims` | `bool` | Keep the reduced dimension after reduction |
| `tie_break_left` | `bool` | If multiple elements have the same minimum value, return the index of the leftmost minimum value |


Return value:  
`tl.tensor`: A tensor with the same shape as `input`.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2A3 Series| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| -------- | ---------------------- |
| GPU    | Unlimited |
| Ascend | Unlimited |


Conclusion: There is no difference between GPU and Ascend platforms in terms of Shape.


### 2.3 Special Limitations


Relative community capability is lacking and cannot be achieved.


Ascend A3 lacks support for uint16, uint32, uint64, and fp64 compared to GPU.


### 2.4 Usage


For more examples, refer to the triton-ascend code repository, ascend/examples/generalization_cases/test_argmin.py.


```python
@triton.jit
def triton_argmin_1d(in_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.argmin(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)
```

