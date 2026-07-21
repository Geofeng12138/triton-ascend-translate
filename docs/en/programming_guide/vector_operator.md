# Vector Operator Development


The Vector operator is primarily executed by the Vector Core, with typical forms including element-wise computation, row-level reduction, type conversion, Gather/Scatter, Mask updates, and small fused operators that do not involve `tl.dot`. The development focus is not on making the grid as fine-grained as possible, but rather on having each program process multiple tiles in a loop within the core, given a fixed number of physical Vector Cores.


## Vector Simple Operator Development


Simple Vector operators can start from the [vector addition example](../examples/01_vector_add_example.md) in this repository or `third_party/ascend/tutorials/01-vector-add.py`. The basic steps for this type of operator are as follows:


1. Use `tl.arange` to construct the contiguous offsets of the current tile.
2. Use `mask` to protect the tail block and prevent out-of-bounds load/store.
3. After completing the element-wise computation, write back the results.
4. When the number of grids is much larger than the number of physical cores, fix the grid to `num_vectorcore` and process in batches within the kernel using `range(pid, num_blocks, num_core)`.


The basic kernel structure is as follows:


```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, num_blocks, num_core):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
```


Prioritize checking three types of issues during development:


- **Data Types**: Ascend Vector units have different support and performance for various integer types. For index, length, and offset data that do not affect precision, prefer using `int32`. Refer to `triton-ascend-ops/tutorial/basic/001-vector_add.zh.md` and `002-vector_cmp.zh.md`.
- **BLOCK_SIZE**: BLOCK_SIZE should be as large as possible within the UB capacity. If UB overflow occurs, first reduce the number of elements processed per iteration, then consider splitting sub-blocks.
- **Number of Cores**: The number of physical Vector Cores on an NPU is typically in the tens. When migrating GPU code with small tiles and large grids to NPU, multiple rounds of dispatch can easily introduce significant overhead.


## Vector Complex Operator Development


Complex Vector operators are typically not single element-wise expressions, but rather combinatorial logic involving discrete memory access, batch rearrangement, multiple outputs, or long hidden sizes. Refer to the following examples in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops):


- [`tutorial/best_practice/004-gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py): Ascend-friendly implementation of Megablocks gather/scatter/scatter_wgrad.
- [`tutorial/best_practice/005-binned_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/005-binned_gather_scatter.py): Gather/scatter after grouping by expert/bin.
- [`tutorial/best_practice/006-padded_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/006-padded_gather_scatter.py): MoE gather/scatter with padding.


The organization of this type of operator is usually:


1. **Partition outer tasks by physical cores**: Use `num_vectorcore` as the grid, with each program responsible for a segment of indices or tokens.
2. **Partition the hidden dimension by UB capacity**: Apply `BLOCK_X` to chunk `NUM_COLUMNS`, reserving space for double buffers, indices, and temporary tensors.
3. **Merge small-grained discrete tasks using `SUB_BLOCK_SIZE`**: Load a group of indices at once, organize them into contiguous temporary blocks in UB, reducing GM scalar memory access and multiple stores.
4. **Manage local data within UB using extended semantics**: First import the extension module via `import triton.language.extra.cann.extension as extension`, then use `extension.insert_slice` to merge multiple rows of data, and use `extension.extract_slice` to extract sub-blocks before scattering writes.
5. **Reserve a unified mask for tail blocks**: In complex gather/scatter operations, index masks, column masks, and expert/bin boundaries coexist. It is recommended to name them separately and combine them only at load/store points.


A typical UB budget approach is as follows:


```python
num_core = get_npu_properties()["num_vectorcore"]
block_size = triton.cdiv(indices_length, num_core)
align_elems = 16
block_x = triton.cdiv(min(num_columns, max_block_x), align_elems) * align_elems
sub_block_size = max((ub_budget - block_x * element_bytes) //
                     (block_x * element_bytes + index_bytes), 1)
```


When the performance of a complex Vector operator does not meet expectations, prioritize troubleshooting from the following directions:


- Whether the grid is much larger than the number of physical Vector Cores, causing multiple rounds of dispatch.
- Whether discrete memory access can be transformed into "batch load into UB and then select within UB".
- Whether the tail axis satisfies 32B alignment; if not, whether transpose or borrowing axis transpose can be used to avoid automatic padding.
- Whether `BLOCK_X` and `SUB_BLOCK_SIZE` cause UB overflow or excessively small transfer granularity.

