# triton.language.rand


## 1. OP Overview


Introduction: Given 1 seed scalar and 1 offset block, returns a random block of type float32 in **U**(**0**,**1**).  
Prototype:


```python
triton.language.rand(
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
| `n_rounds`    | `constexpr`, default value is 10 | Number of rounds for the Philox algorithm |


Return value:  
A random block of type `float32`, with the same shape as `offset`, uniformly distributed in the interval `[0.0, 1.0)`.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


Input seed type:


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √    | √     | √     |√     | ×    | ×    | ×    | ×    | √    |


#### 2.2.2 Shape Support


No special requirements.


### 2.3 Special Restrictions


Community capabilities are not currently supported.


### 2.4 Usage


The following example implements a call to `rand`:


```python
@triton.jit
def kernel_rand(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.rand(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals) # 存储随机数

y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()
numel = y_calf.numel()
ncore = 1 if numel < 32 else 32
xblock = math.ceil(numel / ncore)
kernel_rand[ncore, 1, 1](y_calf, 10, numel, xblock)
```

