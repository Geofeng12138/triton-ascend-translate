# triton.language.sort


## 1. Function Overview


Introduction: Sorts the input tensor `x` in ascending or descending order along a specified dimension.


```python
triton.language.sort(x, dim: constexpr | None = None, descending: constexpr = False)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter Name | Type     | Description                |
|---------------|----------|----------------------------|
| `x`           | `tensor` | Tensor data                |
| `dim`         | `int`    | Sorting dimension          |
| `descending`  | `bool`   | Whether to sort descending |


Return value:  
`x`: The shape of the output tensor is the same as the shape of the input `x`.


### 2.2 OP Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √      | √     | ×     | ×      | ×      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √     | √      | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |


Conclusion: Ascend lacks support for int32, uint8, int64, float64, and bool compared to GPU.  
torch_npu supports uint8.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensor |
| Ascend A2/A3 | Only supports 1~5D tensor |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms, both support 1 to 5-dimensional tensors.


### 2.3 Special Restrictions Explanation


Relative community capability deficiency and inability to achieve


Due to limitations of the Bisheng compiler, int32, uint8, int64, float64, and bool cannot be implemented.


### 2.4 Usage


The following example demonstrates sorting the input tensor `x`:


```python
@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    offx = tl.arange(0, M)
    offy = pid * M
    off2d = offx + offy
    x = tl.load(X + off2d)
    x = tl.sort(x, dim=0, descending=descending)
    tl.store(Z + off2d, x)
```

