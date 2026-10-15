# triton.language.extract_slice


## 1. OP Overview


Introduction: Extract a tensor from the input tensor according to the offset, size, and stride parameters specified by the operation.  
Prototype:


```python
triton.language.extract_slice(
    ful,
    offsets,
    sizes,
    strides,
    _builder=None,
    _generator=None
) -> tensor
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type             | Description                                                    |
| ------------- | ---------------- | -------------------------------------------------------------- |
| `ful`         | `tensor`         | The source tensor from which to extract the slice              |
| `offsets`     | `tuple of ints`  | Starting offsets of the slice along each dimension             |
| `sizes`       | `tuple of ints`  | Sizes of the slice along each dimension                        |
| `strides`     | `tuple of ints`  | Strides of the slice along each dimension                      |
| `_builder`    | -                | Reserved parameter, external calls not supported currently     |
| `_generator`  | -                | Reserved parameter, external calls not supported currently     |


Return value:  
`tensor`: the extracted slice tensor


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | ×    |


#### 2.2.2 Shape Support


Supports tensors of arbitrary shape, but the slice size must not exceed the size of the corresponding dimension of the source tensor.


### 2.3 Special Limitations


No special restrictions.


### 2.4 Usage


The following example extracts the first 32 elements from the computation result:


```python
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 提取前32个元素
    out_sub = tl.extract_slice(output, [block_start], [32], [1])
    out_idx = block_start + tl.arange(0, 32)
    out_msk = out_idx < n_elements
    tl.store(output_ptr + out_idx, out_sub, mask=out_msk)
```

