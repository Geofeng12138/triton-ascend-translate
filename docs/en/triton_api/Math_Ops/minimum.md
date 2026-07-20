# triton.language.minimum


## 1. Function Overview


Introduction: Compute the element-wise minimum of x and y.


```python
triton.language.minimum(x, y, propagate_nan: ~triton.language.core.constexpr = <PROPAGATE_NAN.NONE: 0>, _semantic=None)¶
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | Tensor data                                                      |
| `y`       | `tensor`    | Tensor data                                                        |
| `propagate_nan`       | `tl.PropagateNan`    | Whether to propagate NaN values                                                        |
| `_semantic`   | -                 | Reserved parameter, external calls are not supported temporarily |


Return value:  
`x`: A tensor with the same shape as the input `x`.


### 2.2 OP Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | ×     | ×     | ×      | ×      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | √     | ×     | ×      | ×      | √     | √    | √    | ×    | √    | √    |


Conclusion: Ascend lacks fp64 support compared to GPU.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensors |
| Ascend A2/A3 | Only supports 1~5D tensors |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms, both supporting 1 to 5-dimensional tensors.


### 2.3 Special Restrictions


Relative community capability is lacking and cannot be achieved.


None.


#### 2.3.1 propagate_nan Parameter Restrictions


**Note: When `propagate_nan=tl.PropagateNAN.NONE`, the system will automatically add NaN value handling logic, which will result in:**


1. **Increased UB Space Usage**: Additional NaN detection and handling require more UB space.  
2. **Potential Performance Degradation**: Due to the added computational logic, operator execution performance may decrease.


**Recommendation:**


- If the input data does not contain NaN values or strict NaN handling semantics are not required, it is recommended to use the default value or select an appropriate `propagate_nan` parameter value based on actual needs.  
- In scenarios where UB space is limited, special attention should be paid to the selection of this parameter to avoid compilation failures caused by insufficient UB space.


### 2.4 Usage


The following example implements the element-wise minimum of input tensors `x` and `y`:


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

    ret = tl.minimum(X, Y)

    tl.store(output_ptr + idx, ret)

```

