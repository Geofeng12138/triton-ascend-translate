# CV Fusion Operator Development


CV fusion operator refers to an operator that simultaneously uses Cube Core and Vector Core within the same operator: Cube Core is typically responsible for `tl.dot`, matrix multiplication, or convolution-like main computations, while Vector Core handles bias, activation, softmax, reduction, mask, layout rearrangement, or cross-block synchronization. The goal of CV fusion is to reduce kernel boundaries and GM round trips, but it requires simultaneous control over Cube tile, Vector tile, UB/L1 occupancy, and synchronization relationships.


## CV Fusion Simple Operator Development


For simple CV fusion, it is recommended to first extract the stable `tl.dot` main computation from the [matrix multiplication example](../examples/05_matrix_multiplication_example.md) in this repository, and then add Vector post-processing before writing back. For more complex slice updates, refer to the [fused attention example](../examples/04_fused_attention_example.md). The minimal path is as follows:


1. First implement a stable Cube main computation, e.g., `acc = tl.dot(a, b, acc)`.
2. Fuse lightweight Vector post-processing before writing back the accumulator, e.g., bias, scale, activation, or dtype cast.
3. For larger accumulators, use `range` with `extension.extract_slice`/`extension.insert_slice` to perform ordinary sub-block partitioning, avoiding UB overflow in the Vector post-processing stage.
4. `extension.parallel(..., bind_sub_block=True)` is a stronger explicit multi-Vector sub-block binding path, which may be unavailable when target hardware and compilation configurations differ; it is not recommended as the default approach for simple examples.


Example structure:


```python
# 在 matmul kernel 内部，K 循环完成后得到 fp32 accumulator。
acc = tl.dot(a, b, acc)  # 通常位于 K 维循环内，这里仅展示结构。

# 在写回前融合轻量 Vector 后处理。
acc = tl.where(acc >= 0, acc, 0.01 * acc)
c = acc.to(tl.float16)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```


When developing simple CV fusion, keep boundaries clear: Cube is responsible for generating larger two-dimensional accumulators, while Vector handles element-wise or small-scale reductions within the same tile. If the Vector part needs to share state across multiple Cube tiles, synchronization, workspace, or kernel splitting must be introduced.


## CV Fusion Complex Operator Development


For complex CV fusion, refer to the best practice in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops).


- [`tutorial/best_practice/002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py): In Decode attention, QK/PV uses Cube, while softmax, mask, exponent, normalization, and discrete KV memory access rearrangement use Vector.
- [`tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md): Demonstrates how to use `extension.insert_slice`, transpose, and kernel splitting optimization to reduce discrete memory access and padding overhead when fusing cat, slice, and conv1d update.


For complex CV fusion, it is recommended to organize by data flow layers.


1. **Main Compute Layer**: Identifies which steps must go through Cube, such as QK, PV, GEMM, and batched matmul.
2. **Vector Post-Processing Layer**: Determines whether operations like softmax, activation, mask, scale, normalization, cat/slice, and layout transform can be completed within the same tile.
3. **Memory Access Reordering Layer**: For discrete KV cache, MoE token reordering, and short-tail axis tensors, prioritize using `extension.insert_slice`, `extension.extract_slice`, transpose, or axis borrowing transpose in UB to form hardware-friendly contiguous access.
4. **Pipeline and Synchronization Layer**: Explores overlapping execution of Cube and Vector through compilation options such as `multibuffer`, `set_workspace_multibuffer`, `tile_mix_vector_loop`, and `tile_mix_cube_loop`.
5. **Kernel Distribution Layer**: CV fusion operators typically launch grids based on the number of Cube cores; at runtime, Vector cores are coordinated at approximately a 1:2 ratio. Do not simply follow the large grid approach used on GPUs.


For attention-based CV fusion, it is recommended to first get the non-causal, short sequence, small head_dim cases working, then gradually add:


- Causal mask processing in stages.
- Long sequence K/V block looping.
- Numerically stable softmax update for `m_i`/`l_i`.
- Accumulator workspace and sub-block partitioning when HEAD_DIM is large.
- Load reordering under discrete indexing of KV cache.


When tuning complex CV fusion, first observe the time proportions of Cube, Vector, and MTE2 in profiling. If Cube is waiting for Vector, consider reducing the granularity of Vector post-processing or enabling CV balance-related options; if Vector is waiting for data movement, prioritize checking discrete memory access, tail-axis padding, and multibuffer configuration.

