# triton.language.atomic_min


## 1. OP Overview


Introduction: Atomic minimum value operation, performs an atomic minimum value operation at the specified memory location.  
Prototype:


```python
triton.language.atomic_min(
    pointer,
    val,
    mask=None,
    sem=None,
    scope=None,
    _semantic=None
) -> pointer
```


Can be called as a member function of a tensor, such as `x.atomic_min(...)`, which is equivalent to `atomic_min(x, ...)`.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ---- | ----------- |
| `pointer` | `triton.PointerDType` | The memory location to operate on; the result of `min(*pointer, val)` is written back to this memory |
| `val` | `pointer.dtype.element_ty` | The value for the atomic min operation (right operand) |
| `mask` | `int1` or `tensor<int1>`, optional | Specifies the data range to prevent out-of-bounds access |
| `sem` | `str`, optional | Specifies the memory semantics of the operation<br>Community official configuration accepts "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE"), and "relaxed"<br>We only support "acq_rel":<br>- acquire: After acquiring the lock, previous release operations are visible (equivalent to a "read" operation that blocks until the "latest" data, i.e., data released by other threads, is readable)<br>- release: All operations before releasing the lock are visible to threads that subsequently acquire the lock (equivalent to a "write" operation that "synchronizes" all previous write operations) |
| `scope` | `str`, optional | The thread scope for observing the synchronization effect of the atomic operation<br>Accepted values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (representing "SYSTEM")<br>We only support "gpu" |
| `_semantic` | - | Reserved parameter; external calls are not currently supported |


Return value:  
`pointer`: tensor, the old value before the operation is performed.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
GPU     | ×     | ×      |  √     | ×     | ×      | ×      | ×      |√    | ×     | √    | ×      | ×      | ×     |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |


Conclusion: Ascend lacks int64 support compared to GPU.


#### 2.2.2 Shape Support


No special requirements


### 2.3 Special Restrictions Explanation


Relative community capability deficiency and inability to achieve


| Difference | Description |
| ----------- | ----------- |
| Data Type | Ascend lacks support for int64 compared to GPU (hardware limitation) |
| sem | The officially accepted values for the community are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE"), and "relaxed".<br>We only support "acq_rel". |
| scope | Accepted values are "gpu", "cta", or "sys".<br>We only support "gpu". |


### 2.4 Usage


The following example implements atomic minimum value calculation:


```python
@triton.jit
def triton_atomic_min(
    in_ptr0, out_ptr0, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_min(out_ptr0 + (x1), tmp0, xmask)
```

