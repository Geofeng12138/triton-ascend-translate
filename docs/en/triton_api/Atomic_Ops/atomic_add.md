# triton.language.atomic_add


## 1. OP Overview


Introduction: Atomic addition operation, performs atomic addition at the specified memory location.  
Prototype:


```python
triton.language.atomic_add(
    pointer,
    val,
    mask=None,
    sem=None,
    scope=None,
    _semantic=None
) -> pointer
```


Can be called as a member function of a tensor, such as `x.atomic_add(...)`, which is equivalent to `atomic_add(x, ...)`.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ---- | ----------- |
| `pointer` | `triton.PointerDType` | The memory location to operate on. The result of the *pointer + val computation is written back to this memory location. |
| `val` | `pointer.dtype.element_ty` | The value for the atomic addition operation (right operand). |
| `mask` | `int1` or `tensor<int1>`, optional | Specifies the data range to prevent out-of-bounds access. |
| `sem` | `str`, optional | Specifies the memory semantics of the operation.<br>Community official configuration accepts "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE"), and "relaxed".<br>We only support "acq_rel":<br>- acquire: After acquiring the lock, previous release operations are visible (equivalent to a "read" operation that blocks until the "latest" data, i.e., data released by other threads, can be read).<br>- release: All operations before releasing the lock are visible to threads that subsequently acquire the lock (equivalent to a "write" operation that "synchronizes" all previous write operations). |
| `scope` | `str`, optional | The thread scope for observing the synchronization effect of the atomic operation.<br>Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (representing "SYSTEM").<br>We only support "gpu". |
| `_semantic` | - | Reserved parameter; external calls are not supported for now. |


Return value:  
`pointer`: tensor, the old value before the operation is performed.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | √    | √    | ×    |
| Ascend A2/A3| √    | √     | √     | √     | √      | √      | ×      | ×     | √    | √    | ×    | √    | ×    |


Conclusion: Ascend lacks support for int64 and fp64 compared to GPU.


#### 2.2.2 Shape Support


No special requirements


### 2.3 Special Limitations


Relative community capabilities are lacking and cannot be achieved.


| Difference | Description |
| ----------- | ----------- |
| Data Type | Ascend lacks support for int64 and fp64 compared to GPU (hardware limitation) |
| sem | The officially accepted values in the community are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE"), and "relaxed".<br>We only support "acq_rel". |
| scope | Accepted values are "gpu", "cta", or "sys".<br>We only support "gpu". |


### 2.4 Usage


The following example implements atomic addition calculation:


```python
@triton.jit
def atomic_add(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_add(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask)

dtype, shape, ncore = ['int16', (32, 32), 2]
block_size = shape[0] * shape[1] / ncore
split_size = shape[0] // ncore
x0_value = 3
x0 = torch.full(shape, x0_value, dtype = getattr(torch, dtype)).npu()
x1 = torch.full((split_size, shape[1]), 2, dtype = getattr(torch, dtype)).npu()
y = torch.full((split_size, shape[1]), -10, dtype = getattr(torch, dtype)).npu()
n_elements = shape[0] * shape[1]
atomic_add[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1])
```

