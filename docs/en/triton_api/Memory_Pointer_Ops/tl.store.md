# triton.language.store


## 1. OP Overview


Prototype:


```python
triton.language.store(
 pointer,
 value,
 mask=None,
 boundary_check=(),
 cache_modifier='',
 eviction_policy='',
 _semantic=None
)
```


Introduction: Store a Tensor/Scalar from UnifiedBuffer back to GlobalMemory at the address pointed to by `pointer`.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ---- | ----------- |
| `pointer` | `triton.PointerType` <br> or `tensor<triton.PointerType>` <br> or `triton.PointerType<tensor>` (from `tl.make_block_ptr`) | Pointer to the address on GM where data is to be stored |
| `value` | `tensor` or `scalar` | Value to be stored, supports implicit broadcasting and implicit type conversion |
| `mask` | `int1` or `tensor<int1>` | Optional parameter, can only be passed when `pointer` is not from `tl.make_block_ptr`<br>If `mask[i]==False`, `value[i]` will not be stored to the address pointed by `pointer[i]`; if `True`, normal storage occurs <br>If `pointer` is from `tl.make_block_ptr`, `mask` must be `None` |
| `boundary_check` | `tuple(int)` | Optional parameter, can only be passed when `pointer` is from `tl.make_block_ptr`<br>Integer tuple indicating dimensions that require boundary checking |
| `cache_modifier` | `""` or `"ca"` or `"cg"` | Optional parameter, controls cache options on NVIDIA PTX, ineffective on Ascend hardware |
| `eviction_policy` | `str` | Controls eviction policy on NVIDIA PTX, ineffective on Ascend hardware |
| `_semantic` | - | Reserved parameter, external calls not supported for now |


Return value: No return value


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    |  √    |


Conclusion: Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).  
Expert opinion: For eviction_policy and cache_modifier, refer to load.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Supports scalar and 1~5D tensors |
| Ascend | Supports scalar and 1~5D tensors |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.


#### 2.2.3 Community Constraints


1. If `pointer` is a single pointer:  
   - In this case, `value` and `mask` must be scalars.  
   - `other` will be implicitly type-cast to the data type of `pointer.dtype.element_ty`.  
   - `boundary_check` is not allowed in this case.  
2. If `pointer` is an N-dimensional tensor:  
   - `mask` and `value` will be implicitly broadcast to the same shape as `pointer`.  
   - `boundary_check` is not allowed in this case.  
3. If `pointer` comes from `tl.make_block_ptr`:  
   - In this case, `mask` must be `None`.  
   - Boundary checking can be set via `boundary_check`.


### 2.3 Special Restrictions


Relative community capabilities are lacking and cannot be achieved.


Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation), and the eviction_policy and cache_modifier functionalities are not yet fully developed on NPU.


| Difference | Description | Solution |
| --- | --- | --- |
| Generalization issue of discrete mask | The current handling of discrete masks in store involves decomposing the store into atomic {load, select, store} operations, which has certain generalization issues in corner cases | Expose issues through extensive generalization testing, resolve iteratively |
| Generalization issue when used with branch and loop statements | If the computation of `pointer` and `mask` in the current `tl.load` involves complex loops and branch statements, compilation issues may arise | Expose issues through extensive generalization testing, resolve iteratively |


### 2.4 Usage


In the following example, the functionality of `torch_ldst_indirect_08_func` is achieved through the coordinated invocation of `triton_ldst_indirect_08_kernel` and `triton_ldst_indirect_08_func`.


```python
@triton.jit
def triton_ldst_indirect_08_kernel(
    out_ptr0, in_ptr1, in_ptr2, in_ptr3, stride_in_r,
    XS: tl.constexpr, RS: tl.constexpr
):
    pid = tl.program_id(0)
    in_idx0 = pid * XS + tl.arange(0, XS)
    in_idx1 = tl.arange(0, RS)
    tmp0 = tl.arange(0, XS)
    tmp1 = tl.load(in_ptr1 + in_idx1)
    in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
    tmp2 = tl.load(in_ptr2 + in_idx2)
    tmp2 = tl_math.exp(tmp2)
    tmp3 = tl.load(in_ptr3 + in_idx1)
    tmp3 = tmp3 + 1 - 8
    out0_idx = in_idx0[:, None] * RS + tmp3[None, :]
    tl.store(out_ptr0 + out0_idx, tmp2)

def triton_ldst_indirect_08_func(xc, x2, xs, rs): # [8-24] ori 8 16
    nr = x2.size()[0]
    nc = xc.numel()
    stride_in_r = x2.stride()[0]
    assert nr == xs, "test only single core"
    y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
    xc1 = xc - 1
    triton_ldst_indirect_08_kernel[nr // xs, 1, 1](
        y0, xc, x2, xc1, stride_in_r, XS = xs, RS = rs)
    return y0

def torch_ldst_indirect_08_func(xr, xc, x2):
    flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
    extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
    print(extracted)
    return torch.exp(extracted)

DEV = "npu"
DTYPE = torch.float32
offset = 8
N0, N1 = 16, 32
blocksize = 8
lowdimsize = N0
assert N1 >= N0+offset, "N1 must be >= N0+offset"
assert N0 == lowdimsize, "N0 must be == lowdimsize"
xc = offset + torch.arange(0, N0, device=DEV)
xr = torch.arange(0, blocksize, device=DEV)
x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
torch_ref = torch_ldst_indirect_08_func(xr, xc, x2)
triton_cal = triton_ldst_indirect_08_func(xc, x2, blocksize, lowdimsize)
torch.testing.assert_close(triton_cal, torch_ref)
```

