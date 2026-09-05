# Cube Operator Development


The Cube operator primarily uses matrix multiplication or batched matrix multiplication as its main computational workload, with `tl.dot` typically at the core in Triton code. The key to the Cube operator is designing tiles around the three dimensions M, N, and K, so that A/B tiles can be efficiently moved on-chip and accumulated on the Cube Core.


## Cube Simple Operator Development


For simple Cube operators, refer to the [matrix multiplication example](../examples/05_matrix_multiplication_example.md) in this repository. A minimal development path includes:


1. Clearly define the input/output shapes and strides, e.g., `A[M, K]`, `B[K, N]`, `C[M, N]`.
2. Use `tl.program_id` to map the current program to the `(pid_m, pid_n)` tile of the output matrix.
3. Construct 2D offsets for A and B using `BLOCK_SIZE_M/N/K`.
4. Loop over the K dimension to load sub-blocks of A and B, and accumulate them into an fp32 accumulator using `tl.dot`.
5. Convert the accumulator to the output dtype and write it back to C with boundary masks.


The core structure is as follows:


```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  stride_am: tl.constexpr, stride_ak: tl.constexpr,
                  stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                  stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```


When tuning parameters for simple Cube operators, prioritize:


- Whether `BLOCK_M/N/K` satisfies hardware support and UB/L1 capacity constraints.
- Whether the K-dimension loop can enable `multibuffer` to form a pipeline of data movement and computation.
- Whether the output tile includes additional bias, scale, or activation. If the post-processing is lightweight, it can still be classified as a Cube operator; if the post-processing involves significant Vector reduction or cross-core synchronization, it should be organized as a CV fusion operator.


## Cube Complex Operator Development


Complex Cube scenarios often arise from attention, batched matmul, grouped matmul, or irregularly shaped matrix multiplications. The complex cases on the current main branch of [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) are concentrated in `tutorial/best_practice/`, where [`002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py) can serve as a reference for the core logic of complex Cube operations: it includes two `tl.dot` segments for QK and PV, and demonstrates how to reorganize K/V memory access under discrete KV cache indexing.


Complex Cube operators should be decomposed in the following order:


1. **First extract the pure matrix multiplication core**: Confirm the input tile shape, dtype, accumulation dtype, and output tile shape for each `tl.dot`.
2. **Then handle irregular memory access**: If the K/V cache has low-dimensional discreteness and high-dimensional continuity, a direct 2D load may degrade to scalar memory access. You can first load into UB along the continuous dimension, then reorganize into the layout required by `tl.dot` through transpose or the Ascend extension interface `extension.insert_slice`.
3. **Leave reduction and normalization to well-defined boundaries**: For example, `max/sum/exp` in attention belong to Vector logic. If placed in the same kernel as `tl.dot`, you need to follow the approach of [CV Fusion Operator Development](./cv_fusion_operator.md).
4. **Design inner loops for long K or long sequences**: The K-dimension loop should control the on-chip occupancy of a single A/B tile; the sequence dimension loop should avoid loading an excessively large K/V block at once.
5. **Use Autotune to manage candidate tiles**: Prepare multiple sets of `BLOCK_M/N/K` and `multibuffer` configurations for common shapes, allowing the runtime to select the optimal combination.


A common risk of complex Cube operators is directly migrating a large number of programs on the GPU to the NPU. If the number of output tiles is much larger than the number of physical Cube cores, consider having each program process multiple tiles through an inner loop, or set `TRITON_ALL_BLOCKS_PARALLEL=1` to reduce scheduling overhead when the logical cores are confirmed to be independent of each other.

