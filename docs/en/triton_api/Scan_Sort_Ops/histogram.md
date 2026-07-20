# triton.language.histogram


## 1. OP Overview


Introduction: Computes a histogram with `num_bins` bins based on `input`, where each bin has a width of 1, starting from 0.  
Prototype:


```python
triton.language.histogram(
 input,
 num_bins,
 mask=None,
 _semantic=None,
 _generator=None
)
```


Can be called as a member function of a tensor, such as `x.histogram(...)`, which is equivalent to `histogram(x, ...)`.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `input`        | `tensor`          | Input data, containing all numerical points whose distribution is to be counted |
| `num_bins`       | `int`    | Defines how many equal-width intervals the entire data range is divided into |
| `mask`     | `int1` or `tensor<int1>`, optional    | Specifies the data range to prevent out-of-bounds access |
| `_semantic`   | -                 | Reserved parameter, external calls not supported for now |
| `_generator` |-                 | Reserved parameter, external calls not supported for now |


Return value:  
A histogram represented by a tensor.  

Note: The current Triton 3.2 version does not yet support mask; support will be added in a future version update. The input range is limited to [0, num_bins-1]; full range support will be added in a future version update.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | √     | ×     | ×      | √      | ×      | ×     | ×    | ×    | ×    | ×    | ×    |
| Ascend A2/A3 | ×    | ×     | √    | ×     | ×      | √      | √      | √     | ×    | ×    | ×    | ×    | ×    |


#### 2.2.2 Shape Support


Currently only one-dimensional is supported.


### 2.3 Special Restrictions Explanation


Relative community capacity deficiency and inability to achieve


### 2.4 Usage


The following example demonstrates the invocation of histogram:


```python
@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)

x = torch.randint(0, N, (M, ), device=device, dtype=torch.int32)
z = torch.empty(N, dtype=torch.int32, device=device)
histogram_kernel[(1, )](x, z, M=M, N=N)
```


## 3. Semantic Gap


Relative community capability is lacking but can develop support

