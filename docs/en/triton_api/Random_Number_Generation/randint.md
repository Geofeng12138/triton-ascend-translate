# triton.language.randint


## 1. OP Overview


Description: Given 1 seed scalar and 1 offset block, returns a random block of type int32.  
Prototype:


```python
triton.language.randint(
 seed,
 offset,
 n_rounds: constexpr = 10
)
```


If multiple streams of random numbers are needed, using `randint4x` may be faster than calling `randint` four times in succession.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ---- | ----------- |
| `seed`        | `int` or `tensor` | Seed used to generate random numbers |
| `offset`      | `int` or `tensor` | Offset used to generate random numbers |
| `n_rounds`    | `constexpr`, default value is 10 | Number of iteration rounds for the Philox algorithm |


Return value:
1 random block of type int32, with the same shape as offset.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


Input seed type:


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √    | √     | √     |√     | ×    | ×    | ×    | ×    | √    |


#### 2.2.2 Shape Support


No special requirements.


### 2.3 Special Restrictions


Relative community capabilities are lacking and cannot be realized.


### 2.4 Usage


The following example implements a call to `randint` (generating a single random number when called):


```python
@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals) # 存储随机数

y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
kernel_randint[ncore, 1, 1](y_cali, 10, numel, xblock)
```

