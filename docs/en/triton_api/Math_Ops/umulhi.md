# triton.language.umulhi


## 1. Function Overview


Introduction: Compute the most significant N bits of each element in the 2N-bit product of x and y.


```python
triton.language.umulhi(x, y, _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter Name | Type     | Description                                      |
|---------------|----------|--------------------------------------------------|
| `x`           | `tensor` | Tensor data                                      |
| `y`           | `tensor` | Tensor data                                      |
| `_semantic`   | -        | Reserved parameter, currently not supported for external calls |


Return value:  
`x`: The shape of the output tensor is the same as that of the input `x`.


### 2.2 OP Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | √     | ×     | ×     | ×      | ×      | √     | ×    | ×    | ×    | ×    | ×    |
| Ascend A2/A3 | ×    | ×     | √     | ×     | ×     | ×      | ×      | ×     | ×    | ×    | ×    | ×    | ×    |


Conclusion: Ascend lacks int64 support compared to GPU.  
torch_npu supports uint8.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensor |
| Ascend A2/A3 | Only supports 1~5D tensor |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms, both supporting tensors from 1 to 5 dimensions.


### 2.3 Special Limitations


int64 is not supported


### 2.4 Usage


The following example implements significant N-bit quantization on the input tensor `x`.


```python
@triton.jit
def umulhi_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = tl.umulhi(x, y)
    tl.store(Z + tl.arange(0, N), z)
```

