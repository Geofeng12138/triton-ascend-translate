# triton.language.semantic.less_equal


## 1. OP Overview


Introduction: Used to compare elements of two tensors, equivalent to `<=`.  
Function prototype:


```python
triton.language.semantic.less_equal(
 input: tl.tensor,
 other: tl.tensor,
 builder: ir.builder
) -> tl.tensor
```


Used as a built-in operator for `tensor`, such as `x<=y`.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| :---: | :---: | :---: |
| `input` | `tensor` | Tensor data, left operand, representing the primary data to be compared |
| `other` | `tensor` | Tensor data, right operand, compared element-wise with `input` |
| `_builder` | - | Reserved parameter, currently not supported for external calls |


Return value:  
`tl.tensor`: A tensor with the same shape as `input`.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | √ | √ | × | √ | √ |


Conclusion: Triton-Ascend currently does not support uint8/uint16/uint32/uint64 and fp64 compared to GPU.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| -------- | ---------------------- |
| GPU    | Unlimited |
| Ascend A2/A3| Unlimited |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms.


### 2.3 Special Restrictions Explanation


> Relative community capability is lacking and cannot be achieved


Triton-Ascend lacks fp64 support compared to GPU, and support for uint8/uint16/uint32/uint64 types is under development.


### 2.4 Usage


The following example implements the less-than-or-equal-to operation on three-dimensional tensors `x0` and `x1`:


```python
@triton.jit
def triton_le_3d(in_ptr0, in_ptr1, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0 <= x1
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)
```

