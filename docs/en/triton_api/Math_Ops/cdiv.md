# triton.language.cdiv


## 1. OP Overview


Introduction: Computes the ceiling division of a tensor.  
Function prototype:


```python
triton.language.cdiv(x, div)
```


Can be called as a member function of a tensor, such as `x.cdiv(...)`, which is equivalent to `cdiv(x, ...)`.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| :---: | :---: | :---: |
| `x` | `tensor` | Tensor data, dividend |
| `div` | `tensor` | Tensor data, divisor |


Return value:  
`out`: A tensor with the same shape as `x` and `div`.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | √ | √ | √ | √ | √ | √ | √ | √ | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | × | × | × | × | × |


Conclusion: Compared with GPU, Ascend does not support uint and bool inputs.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| -------- | ---------------------- |
| GPU    | Unlimited |
| Ascend | Unlimited |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms.


### 2.3 Special Restrictions


Relative community capabilities are lacking and cannot be achieved.


Input range: 0~16777216


### 2.4 Usage


The following example implements ceiling division on input tensors `x` and `y`:


```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.cdiv(X, Y)

    tl.store(output_ptr + idx, ret)
```

