# triton.language.reduce


## 1. OP Overview


Introduction: `triton.language.reduce` applies `combine_fn` along the specified axis `axis` to reduce the input tensor, returning the resulting tensor after reduction.


```python
triton.language.reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None)
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
|--------|------|------|
| `input` | `Tensor` or `tuple of Tensor` | Input tensor, can be a single tensor or a tuple of tensors |
| `axis` | `int` or `None` | The axis along which to perform the reduce operation. If None, all axes are reduced |
| `combine_fn` | `Callable` | Function used to combine two groups of scalar tensors (must be decorated with `@triton.jit`) |
| `keep_dims` | `bool` | If True, keeps the reduced axes with length 1 |
| `_semantic` | `Optional[str]` | Reserved parameter, external calls not supported for now |
| `_generator` | `Optional[Generator]` | Reserved parameter, external calls not supported for now |


**Note**: This function can also be called as a member function of a tensor, such as `x.reduce(...)` instead of `reduce(x, ...)`.


Return value:  
`tensor`: The resulting tensor after reduction along the specified axis `axis`.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


#### 2.2.2 Shape Support


Conclusion: There is no difference between GPU and Ascend platforms in terms of Shape.


### 2.3 Special Restrictions Explanation


Relative community capability is missing and cannot be achieved.  
`keep_dims=True` requires testing on more specifications to determine whether it is fully supported. Currently, it has been tested that in the 3D `dim=2` case, `keep_dims=True` is supported.


### 2.4 Usage


The following example implements reduce computation on a 2D shape tensor, where the combine_fn uses simple addition:


```python
@triton.jit
def _reduce_combine(a, b):
    return a + b

@triton.jit
def tt_reduce_2d(in_ptr, out_ptr,
                 xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                 XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)

    if dim == 0:
        oidx = yidx
    else:
        oidx = xidx
    tl.store(out_ptr + oidx, ret)

```

