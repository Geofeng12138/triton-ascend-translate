# triton.language.core.__rshift__


## 1. OP Overview


Description: Right-shift the tensor according to the given value.


```python
triton.language.core.__rshift__(
 input: tl.tensor,
 other: tl.tensor,
 builder: ir.builder
) -> tl.tensor
```


Used as a built-in operator for `tensor`, such as `x>>y`.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| :---: | :---: | --- |
| `input` | `tensor` | Tensor data, the left operand, representing the primary data to be shifted |
| `other` | `tensor or scalar` | Tensor data, the right operand, the value by which to shift |
| `_builder` | - | Reserved parameter, external calls not supported for now |


Return value:  
`tl.tensor`: A tensor with the same shape as `input`.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPU      | √ | √ | √ | √ | √ | √ | √ | √ | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | × | × | × | × | √ |


Conclusion: Ascend lacks support for uint compared to GPU.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
|--------|--------------------------|
| GPU    | Unlimited                |
| Ascend A2/A3 | Unlimited                |


Conclusion: There is no difference between GPU and Ascend platform in terms of Shape.


### 2.3 Special Restrictions Explanation


Relative community capability is lacking and cannot be achieved.


1. Ascend lacks support for the `uint` type compared to GPU.  
2. The right operand `other` only supports scalars, not tensors (i.e., `x >> 2` is valid, while `x >> y` (where `y` is a tensor) is not yet supported).


### 2.4 Usage


The following example implements right shift operation on three-dimensional tensors `x0` and `x1`:


```python
@triton.jit
def triton_lshift_3d(in_ptr0, out_ptr0, L : tl.constexpr, M : tl.constexpr, N : tl.constexpr):
    loffs = tl.program_id(0) * L
    lblk_idx = tl.arange(0,L) + loffs
    mblk_idx = tl.arange(0,M)
    nblk_idx = tl.arange(0,N)
    idx = lblk_idx[:,None,None]*N*M+mblk_idx[None,:,None]*N+nblk_idx[None,None,:]
    x0=tl.load(in_ptr0+idx)
    ret = x0 >> 2
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0+odx, ret)
```

