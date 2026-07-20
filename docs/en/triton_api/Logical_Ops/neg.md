# triton.language.tensor.__neg__


## 1. Function Overview


Introduction: Negate the value of a tensor.


```python
# 通过操作符
-x

# 或直接调用 dunder 方法
x.__neg__()
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | Tensor data |
| `_semantic`   | -                 | Reserved parameter, currently not supported for external calls |


Return value:  
`out`: The shape of the output tensor is the same as the shape of the input `x`.


### 2.2 OP Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √     | √      | √     | √      |  √      |  √       |  √       | √      | √    | √   | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | √      | ×     | ×      | ×      | √     | √    | √    | ×    | √    | ×   |


Conclusion: Ascend lacks support for uint, fp64, and bool types compared to GPU.


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensor |
| Ascend A2/A3 | Only supports 1~5D tensor |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms, both supporting 1 to 5 dimensional tensors.


### 2.3 Special Restrictions Explanation


Relative community capability deficiency and inability to achieve


Ascend lacks support for uint, fp64, and bool types compared to GPU.


### 2.4 Usage


The following example implements element-wise negation of the input tensor `x`:


```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = -X

    tl.store(output_ptr + idx, ret)

x = test_common.generate_tensor(shape, dtype).npu()
```

