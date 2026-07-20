# triton.language.randint4x


## 1. OP Overview


Introduction: Given 1 seed scalar and 1 offset block, returns 4 random blocks of type int32.  
The most efficient entry point of Triton's Philox pseudo-random number generator.  
Prototype:


```python
triton.language.randint4x(
 seed,
 offset,
 n_rounds: constexpr = 10
)
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `seed`        | `int` or `tensor` | Seed used for generating random numbers |
| `offset`      | `int` or `tensor` | Offset used for generating random numbers |
| `n_rounds`    | `constexpr`, default value is 10 | Number of iteration rounds for the Philox algorithm |


Return value:  
4 random blocks of type int32, each block has the same shape as `offset`.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


Input seed type:


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √    | √     | √     |√     | ×    | ×    | ×    | ×    | √    |


#### 2.2.2 Shape Support


No special requirements.


### 2.3 Special Restrictions


Relatively lacking in community capabilities and unable to achieve


### 2.4 Usage


The following example demonstrates the call to `randint4x` when the offset is a scalar:


```python
@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        mask = (global_offset + indices) < N
        tl.store(x_ptr + global_offset + indices, rand_vals, mask) # 存储随机数

y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)
```


The following example implements a call to `randint4x` when the offset is non-scalar, where the tensor used for storage is 4 times the size of the offset:


```python
@triton.jit
def triton_randint4x1d(out_ptr, seed, L: tl.constexpr):
 idx = tl.arange(0, L)
 rnd0, rnd1, rnd2, rnd3 = tl.randint4x(seed, idx)
 pointers0 = out_ptr + idx
 pointers1 = out_ptr + L + idx
 pointers2 = out_ptr + 2 * L + idx
 pointers3 = out_ptr + 3 * L + idx
 tl.store(pointers0, rnd0)
 tl.store(pointers1, rnd1)
 tl.store(pointers2, rnd2)
 tl.store(pointers3, rnd3)
```

