# triton.language.softmax


## 1. Function Overview


Introduction: Compute the element-wise softmax of x.


```python
triton.language.softmax(x, dim=None, keep_dims=False, ieee_rounding=False)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter Name | Type               | Description                                                    |
| ------------- | ------------------ | -------------------------------------------------------------- |
| `x`           | `tensor`           | Tensor data                                                    |
| `dim`         | `int`              | Specifies the dimension on which to compute softmax            |
| `keep_dims`   | `bool`             | Controls whether to retain the original dimension shape after computation |
| `ieee_rounding` | `bool`           | Controls whether floating-point operations follow IEEE 754 rounding rules |


Return value:  
`x`: A tensor with the same shape as `x`.


### 2.2 OP Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     |  √    | √    | √    |  √    | ×    |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |


Conclusion: Ascend lacks fp64 support compared to GPU.
torch_npu does not support uint8.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensor |
| Ascend A2/A3 | Only supports 1~5D tensor |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.


### 2.3 Special Limitations


Relative community capability is lacking and cannot be achieved.


No.


### 2.4 Usage


The following example implements element-wise softmax on the input tensor `x`:


```python
@triton.jit
def tt_softmax_3d(in_ptr, out_ptr,
                  xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                  XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * ynumel * znumel + yidx[None, :, None] * znumel + zidx[None, None, :]

    a = tl.load(in_ptr + idx)
    ret = tl.softmax(a, dim=2)

    tl.store(out_ptr + idx, ret)
```

