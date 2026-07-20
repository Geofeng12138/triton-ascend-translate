# triton.language.insert_slice


## 1. OP Overview


Introduction: Insert a tensor (sub-tensor) into a specified position of another tensor, that is, insert one tensor into another tensor according to the offset, size, and stride parameters specified by the operation.
Prototype:


```python
triton.language.insert_slice(
    ful,
    sub,
    offsets,
    sizes,
    strides,
    _builder=None,
    _generator=None
) -> tensor
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `ful` | `tensor` | The target tensor to receive the insertion |
| `sub` | `tensor` | The sub-tensor to be inserted, whose shape must match the shape specified by the `sizes` parameter |
| `offsets` | `tuple of ints` | Specifies the starting offsets (per dimension) for insertion into the `ful` tensor |
| `sizes` | `tuple of ints` | Specifies the size of the insertion region (per dimension) |
| `strides` | `tuple of ints` | Specifies the stride of the insertion region (per dimension) |
| `_builder` | - | Reserved parameter, external calls not supported for now |
| `_generator` | - | Reserved parameter, external calls not supported for now |


Return value:  
`tensor`: a new tensor after inserting the sub-tensor


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | ×    |


#### 2.2.2 Shape Support


Supports tensors of arbitrary shapes, provided that:


1. The number of dimensions of `ful` and `sub` must be the same.
2. The lengths of `offsets`, `sizes`, and `strides` must be the same as the number of tensor dimensions.
3. The insertion region must not exceed the boundaries of the `ful` tensor.


### 2.3 Special Limitations Explanation


No special restrictions


### 2.4 Usage


The following example demonstrates inserting the slice computation result back into the original tensor:


```python
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # 提取切片
    x_sub = tl.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = tl.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    # 加载原始输出张量
    output = tl.load(output_ptr + offsets, mask=mask)
    # 将计算结果插入回原张量
    output = tl.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
```


## 3. Semantic Gap


No semantic difference.

