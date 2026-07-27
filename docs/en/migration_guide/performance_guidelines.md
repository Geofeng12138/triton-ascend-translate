# NPU High-Performance Programming Guide


## Merge Grid Core Splitting


### 1. Principles of Automatic Merging Grid Core Optimization


In some scenarios, Triton operators are migrated from GPU to NPU. Due to architectural differences, Triton operators developed on GPU have a large number of grid blocks. When executed on NPU, they cannot be fully scheduled in one go, and multiple rounds of dispatch lead to excessive dispatch latency, affecting operator performance. During the optimization of Triton operators on NPU, it is necessary to first check the number of grid blocks. When the number of blocks is large, use the `TRITON_ALL_BLOCKS_PARALLEL` environment variable to improve operator execution performance.


## Instruction Parallelism Optimization


### 1. Core Principles of Instruction Parallelism Optimization


When Triton operators execute on the NPU, to improve performance, the NPU underlying layer provides parallel mechanisms such as multi-buffer and instruction parallelism, which parallelize "data loading / data computation / data offloading" to enhance performance. However, in certain scenarios, multi-buffer may fail to be enabled, affecting parallelism and leading to reduced operator execution performance. During performance optimization, if such issues arise, refer to the following points for troubleshooting and optimize according to the code examples:

1. Data loading and computation have data dependencies, causing synchronization. The MTE loading can only be triggered after Vector computation is completed, resulting in low parallelism.
2. Within the operator, there are no multiple data loads, or a single execution completes without Tiling partitioning. In such scenarios, multi-buffer cannot be enabled.
3. Multi-buffer requires additional UB space usage. If there is insufficient UB space during computation, multi-buffer cannot be enabled.


### 2. Code Examples


- Example 1: Reduce synchronization to improve parallelism


During operator tuning, increasing instruction-level parallelism is an important approach. In the following `tl.load` statement, when N > M, the loaded data can only fill part of the tensor memory space pointed to by `data`. For the remaining unfilled portion, if the user does not specify an `other` value, the GPU defaults to filling it with 0. To reduce the adaptation effort for user migration, the NPU maintains behavior consistent with the GPU. The NPU first uses the Vector core to set the entire memory space pointed to by `data` to the specified value (if the user does not specify an `other` value, it is also set to 0), and then uses the MTE2 instruction to transfer data to the partial memory space pointed to by `data`. This creates a dependency between MTE2 and Vector, preventing efficient parallelism and impacting performance.


    ```diff
    @triton.jit
    def npu_vector_add_kernel(
        input,                          # [Tensor] input tensor (1 x col)
        output,                         # [Tensor] output tensor (1 x col)
        M: tl.constexpr,                # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        N :tl.constexpr = BLOCK_SIZE
        idx = tl.arange(0, N)
        mask = idx < M
        data = tl.load(input + idx, mask = mask) # 或者指定other=-1等值
    ```


To improve performance, when a load operation only partially fills the target memory space and the unfilled portion does not affect subsequent computation results, you can add `care_padding=False` to the load statement to skip default value padding, thereby increasing parallelism and enhancing performance. The optimized version of the above operator is as follows:


    ```diff
    @triton.jit
    def npu_vector_add_kernel(
        input,                          # [Tensor] input tensor (1 x col)
        output,                         # [Tensor] output tensor (1 x col)
        M: tl.constexpr,                # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.arange(0, N)
        mask = idx < M
    -   data = tl.load(input + idx, mask = mask) # 或者指定other=-1等值
    +   data = tl.load(input + idx, mask = mask, care_padding=False) # 或者指定other=-1等值
    ```


- Example 2: Within the Triton operator, use a for loop to add Tiling and improve parallelism.


In Triton operator programming, `mask` operations are often used in syntax such as load/store/where. During performance optimization, special attention must be paid to the performance degradation caused by such operations. When the logic within a Triton operator is a single sequential execution — start → data load → computation → data store → end — instructions cannot be parallelized, resulting in low execution efficiency. By using a `for` loop to add tiling in the operator, the single processing volume can be reduced and processed in multiple batches, allowing "data load / computation / data store" to run in parallel, reducing serial waiting time and improving overall performance. Additionally, using a `for` loop to add tiling also reduces the UB space consumed per single processing step.  
Note: When increasing data tiling, it is also necessary to consider whether the mathematics after changing the data partitioning remains equivalent.


    ```diff
    @triton.jit
    def alloc_extend_kernel(
            pre_lens_ptr,
            seq_lens_ptr,
            free_page_ptr,
            out_indices,
            bs_upper: tl.constexpr,
            page_size: tl.constexpr,
            max_num_extend_tokens: tl.constexpr,
    +       BLOCK_SIZE: tl.constexpr = 1024,
    ):
        pid = tl.program_id(0)

        load_offset = tl.arange(0, bs_upper)
        seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
        pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
        extend_lens = seq_lens - pre_lens

        seq_len = tl.load(seq_lens_ptr + pid)
        pre_len = tl.load(pre_lens_ptr + pid)
        extend_len = seq_len - pre_len

        sum_extend_lens = tl.sum(extend_lens)
        output_start_loc = sum_extend_lens - extend_len

        num_pages_after = (seq_lens + page_size - 1) // page_size
        num_pages_before = (pre_lens + page_size - 1) // page_size
        num_new_pages = num_pages_after - num_pages_before

        num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
                pre_len + page_size - 1
        ) // page_size
        sum_num_new_pages = tl.sum(num_new_pages)
        new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

        # Part 2: fill the new full pages
        num_part2 = (
                seq_len // page_size * page_size
                - (pre_len + page_size - 1) // page_size * page_size
        )

    -   # load data at once
    -   offset_many_page = tl.arange(0, max_num_extend_tokens)
    -   page_start = tl.load(
    -       free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    -       mask=offset_many_page < num_part2,
    -   )
    -   tl.store(
    -       out_indices + output_start_loc + offset_many_page,
    -       page_start * page_size + offset_many_page % page_size,
    -       mask=offset_many_page < num_part2,
    -   )

    +   # load data using loop
    +   num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    +   blk_offset = tl.arange(0, BLOCK_SIZE)
    +   for i in range(num_loop):
    +       offset_many_page = blk_offset + i * BLOCK_SIZE
    +       page_start = tl.load(
    +           free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    +           mask=offset_many_page < num_part2,
    +       )
    +       tl.store(
    +           out_indices + output_start_loc + offset_many_page,
    +           page_start * page_size + offset_many_page % page_size,
    +           mask=offset_many_page < num_part2,
    +       )
    ```


## Data Type Optimization


### 1. Core Principles of Data Type Optimization


The A2/A3 vector arithmetic unit does not support certain data types for some operations. In such cases, the corresponding vector operations degrade to scalar operations, impacting performance. If it does not affect the overall operator precision, it is recommended to use supported data types to improve performance. The following operations are primarily involved.


| **OP Name** | **Unsupported Data Types** |
|---|---|
| Vector ADD | int64 |
| Vector CMP | int64/int32 |


### 2. Code Examples


- Vector Add Triton Operator Example Code


For the following Triton operator, when the data type used by the x and y input tensors is int64, the x1 + y1 operation will be expanded into scalar operations, reducing performance. Without affecting precision, it is recommended to use the int32 data type.


    ``` diff
    @triton.jit
    def npu_vector_add_kernel(
        x,                          # [Tensor] input tensor (1 x col)
        y,                          # [Tensor] input tensor (1 x col)
        z,                          # [Tensor] output tensor (1 x col)
        vector_len: tl.constexpr,   # len of the vector
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(BLOCK_SIZE)
        len_mask = offset < vector_len
        x1 = tl.load(x + offset, mask=len_mask)
        y1 = tl.load(y + offset, mask=len_mask)
        z1 = x1 + y1
        tl.store(z + offset, z1, mask=len_mask)
    ```


- Vector Cmp Triton operator example code


For the following Triton operator, the Cmp operation is used during mask computation. Cmp does not support int64/int32 data types, which causes the `cols < N` operation to be expanded into scalar operations, degrading performance. Without affecting precision, it is recommended to use the fp32 data type.  
In Triton operator programming, mask operations are frequently used in syntax such as load/store/where. During performance optimization, special attention should be paid to the performance degradation caused by such operations.


    ``` diff
    @triton.jit
    def npu_vector_cmp_kernel(
        X,                 # [Tensor] input tensor (row x col)
        Out,               # [Tensor] output tensor (row x col)
        Mean,              # [Vector] mean tensor (row, ) of X
        Rstd,              # [Vector] std tensor (row, ) of X
        stride_x_row,      # [Scalar] stride of row of x
        stride_out_row,    # [Scalar] stride of row of out, normally equals to stride_x_row
        M,                 # [Scalar] row number
        N,                 # [Scalar] col number
        eps,               # [Scalar] epsilon to avoid division by zeros
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        group_m = tl.program_id(0)
        group_n = tl.program_id(1)
        row = group_m

        # calculate index & offset
        Mean = Mean + group_n * M
        Rstd = Rstd + group_n * M
        X = X + row * stride_x_row + group_n * N
        Out = Out + row * stride_out_row + group_n * N

        cols = tl.arange(0, BLOCK_N)  # cols is int64
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

        # calculate mean & rstd
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        # [Changed begin]
    -   xbar = tl.where(cols < N, X - mean, 0.0)
    +   cols_cmp = cols.to(tl.float32)
    +   xbar = tl.where(cols_cmp < N, x - mean, 0.0)
        # [Changed end]

        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)

        # calculate Out
        mask = cols < N
        out = (x - mean) * rstd
        tl.store(Out + cols, out, mask=mask)
    ```

