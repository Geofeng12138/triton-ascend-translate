# triton.language.clamp


## 1. Function Overview


Introduction: Limits the range of tensor x to between [min, max].


```python
triton.language.clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`           | `tensor`          | Tensor data |
| `min`         | `tensor`          | Lower bound (can be a tensor or scalar, broadcast to the shape of `x`) |
| `max`         | `tensor`          | Upper bound (can be a tensor or scalar, broadcast to the shape of `x`) |
| `propagate_nan` | `triton.language.core.constexpr` | Whether to propagate NaN for min or max |
| `_semantic`   | -                 | Reserved parameter, currently not supported for external calls |


Return value:  
`x`: The shape of the output tensor is the same as the shape of the input `x`.


### 2.2 OP Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | √    | √    | ×    |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensor |
| Ascend | Only supports 1~5D tensor |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.


### 2.3 Special Restrictions Explanation


Relative community capability is lacking and cannot be achieved.


Ascend lacks fp64 support compared to GPU.


#### 2.3.1 propagate_nan Parameter Restrictions


**Note: When `propagate_nan=tl.PropagateNAN.NONE`, the system will automatically add NaN value handling logic, which will result in:**


1. **Increased UB Space Usage**: Additional NaN detection and handling require more UB space.  
2. **Potential Performance Degradation**: Due to the added computational logic, operator execution performance may decrease.


**Recommendation:**


- If the input data does not contain NaN values or does not require strict NaN handling semantics, it is recommended to use the default value or select an appropriate `propagate_nan` parameter value based on actual requirements.  
- In scenarios where UB space is limited, special attention should be paid to the selection of this parameter to avoid compilation failures caused by insufficient UB space.


### 2.4 Usage


The following example implements truncation computation on the input tensor `x`:


```python
@triton.jit
def tt_clamp_2d(in_ptr, out_ptr, min_ptr, max_ptr,
                   xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                   XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
       xoffs = tl.program_id(0) * XB
       yoffs = tl.program_id(1) * YB
       xidx = tl.arange(0, XB) + xoffs
       yidx = tl.arange(0, YB) + yoffs
       idx = xidx[:, None] * ynumel + yidx[None, :]

       x = tl.load(in_ptr + idx)
       min_ = tl.load(min_ptr + idx)
       max_ = tl.load(max_ptr + idx)
       ret = tl.clamp(x, min_, max_)

       tl.store(out_ptr + idx, ret)
```

