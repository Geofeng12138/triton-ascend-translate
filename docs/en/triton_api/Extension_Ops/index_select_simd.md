# triton.language.extra.ascend.libdevice.index_select_simd


## 1 Function Description


Parallel gather multiple indices on non-tail axis dimensions, and directly copy data from global memory (GM) to the correct positions in the unified buffer (UB) in units of tiles with zero-copy. This operation is equivalent to a high-performance implementation of `torch.index_select`, suitable for scenarios such as embedding layer lookup and sparse index access.


**Syntax:**


- `triton.language.extra.ascend.libdevice.index_select_simd(src, dim, index, src_shape, src_offset, read_shape)`


**Function:**


- On the specified dimension of the source tensor, batch read data according to the index array
- Support specifying the offset and size of the read region for flexible slicing
- Zero-copy efficient implementation, directly moving data from GM to UB
- Keep the element type and encoding unchanged


**Typical Application Scenarios:**


- Embedding layer lookup: batch reading word vectors from a large vocabulary based on token IDs
- Sparse tensor operations: accessing specific rows of a dense tensor based on sparse indices
- Dynamic routing and attention mechanisms: selecting specific features based on dynamically computed indices


## 2 Parameter Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Required | Description |
|--------|------|------|------|
| src | tensor/pointer | Yes | Source tensor pointer, data located in global memory (GM) |
| dim | int | Yes | The dimension on which to perform the index_select operation, value range [0, len(src_shape)-2], **does not support the trailing axis** (the last dimension) |
| index | tensor | Yes | 1D index array, located on UB, specifying the index positions to read |
| src_shape | Tuple[int] | Yes | The full shape of the source tensor |
| src_offset | Tuple[int] | Yes | The starting position for reading; can be set to -1 on the dim dimension (this dimension is determined by index) |
| read_shape | Tuple[int] | Yes | The size of data to read; must be set to -1 on the dim dimension (this dimension is determined by the length of index) |


**Return value:**


- **Type:** tensor (located on UB)
- **Shape:** consistent with read_shape, where the size of the dim dimension equals the length of index
- **Data Type:** same as the source tensor
- **Memory Location:** Unified Buffer (UB)


**Constraints:**


- `read_shape[dim]` must be -1
- `src_offset[dim]` can be set to -1 (will be ignored because this dimension is determined by index)
- `len(src_shape) == len(src_offset) == len(read_shape)`
- `index` must be a 1D tensor
- `dim` cannot be the trailing axis (last dimension), i.e., `dim < len(src_shape) - 1`
- For non-dim dimensions: `0 <= src_offset[i] < src_shape[i]`
- For non-dim dimensions: `src_offset[i] + read_shape[i] <= src_shape[i]` (out-of-bounds will be automatically truncated)
- Index values in index must be within the range `[0, src_shape[dim])`


### 2.2 DataType Support Table


| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|---------------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |


**Note:**


- The data type of index must be int32 or int64
- This operation is not supported on the GPU platform (Ascend-specific intrinsic)


### 2.3 Shape Support Table


Supports any number of dimensions (from 1D to high-dimensional tensors), subject to the following conditions:


- `index` must be a 1D tensor
- The size of each dimension of the source tensor must be limited by the actual hardware memory
- The size of `read_shape` on non-`dim` dimensions must consider the UB space constraints


**Common Shape Combinations:**


- 2D tensors: suitable for embedding layer lookups and sparse matrix row selection
- 3D tensors: suitable for batch embedding lookups and sequence feature extraction
- High-dimensional tensors: suitable for complex multi-dimensional indexing operations


### 2.4 Special Limitations Explanation


1. **dim restriction:** The `index_select` operation is not supported on the trailing axis (the last dimension); `dim` must satisfy `dim < len(src_shape) - 1`.
2. **Data type restriction:** The uint16/uint32/uint64/float8/float64 data types are currently not supported.
3. **Index out of bounds:** Whether the indices in `index` are out of bounds is not checked; users must ensure the validity of the indices themselves.


### 2.5 Usage


**Basic Usage (2D Embedding Lookup):**


```python
import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice

@triton.jit
def embedding_kernel(
    embed_ptr,      # [vocab_size, embed_dim]
    indices_ptr,    # [batch_size]
    output_ptr,     # [batch_size, embed_dim]
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
):
    pid = tl.program_id(0)

    # 加载索引
    indices = tl.load(indices_ptr + pid * 16 + tl.arange(0, 16))

    # 使用 index_select 批量读取嵌入向量
    embeddings = libdevice.index_select_simd(
        src=embed_ptr,
        dim=0,
        index=indices,
        src_shape=(vocab_size, embed_dim),
        src_offset=(-1, 0),
        read_shape=(-1, embed_dim)
    )

    # 存储结果
    offsets = tl.arange(0, 16)[:, None] * embed_dim + tl.arange(0, embed_dim)[None, :]
    tl.store(output_ptr + pid * 16 * embed_dim + offsets, embeddings)
```


**Relationship with torch.index_select:**


- `index_select_simd` is equivalent to `torch.index_select(src, dim, index)` combined with a slicing operation.
- However, `index_select_simd` is implemented at the hardware level, offering better performance than the PyTorch implementation (approximately 0.6~1.5x the performance of AscendC).


**Differences from regular load:**


```python
## 常规 load 方式（低效）
for i in range(len(indices)):
    idx = tl.load(indices_ptr + i)
    offsets = idx * stride + tl.arange(0, size)
    data = tl.load(src_ptr + offsets)
    # ... 处理 data

## index_select 方式（高效）
indices = tl.load(indices_ptr + tl.arange(0, len(indices)))
data = libdevice.index_select_simd(
    src=src_ptr,
    dim=0,
    index=indices,
    src_shape=(...),
    src_offset=(-1, 0),
    read_shape=(-1, size)
)
## 一次性获取所有数据
```


## 3 Differences from GPU


New OP, no difference


## 4 Test Case Description


**Test file:**


- `ascend/examples/pytest_ut/test_index_select.py` - 2D tensor index_select test (multiple shape combinations)

