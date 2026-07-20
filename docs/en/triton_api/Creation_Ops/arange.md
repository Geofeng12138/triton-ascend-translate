# triton.language.arange


## 1. OP Overview


Introduction: The `triton.language.arange` function is used to generate a contiguous sequence of integers from `start` to `end` (excluding `end`).


```python
triton.language.arange(start, end, _semantic=None)
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type    | Description                                                  |
| -------------- | ------- | ------------------------------------------------------------ |
| `start`        | `scalar` | The starting value for creating a sequence of consecutive integers, must be a compile-time constant (`tl.constexpr`) |
| `end`          | `scalar` | The ending value for creating a sequence of consecutive integers |


Return value:  
`tensor`: a tensor of consecutive integer sequences


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


Conclusion: The parameters `start` and `end` of `arange` must be constants, hence they are untyped. They support the value range corresponding to the type, up to `int32`, and the hardware instructions also only support up to `int32`.


|                | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| --------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| GPU            | ×    | ×     | ×     | ×     | ×      | √      | ×      | ×     | ×    | ×    | ×    |   ×    |
| Ascend A2/A3 | ×    | ×     | √     | ×     | ×      | ×      | ×      | ×     | ×    | ×    | ×    |   ×    |


#### 2.2.2 Shape Support


0 =< (end - start) < 1048576
end >= 0, start >= 0


Conclusion: There is no difference between GPU and Ascend platforms in terms of Shape.


### 2.3 Special Limitations


Relative community capability is lacking and cannot be achieved.


1. The function is used to generate a continuous integer sequence in the range [start, end). CUDA requires that range=(end-start) must be a power of 2. Triton-ascend does not have this requirement.  
2. Both NV and Triton-ascend limit the maximum value of end to TRITON_MAX_TENSOR_NUMEL = 1048576.  
3. The input of arange must be a constant, supporting uint and int type values less than 1048576 (the maximum value TRITON_MAX_TENSOR_NUMEL). int64 is not supported.  
4. The start and end of arange should be greater than or equal to 0.


### 2.4 Usage


The following example implements generating a continuous integer sequence from [0, 128):


```python
@triton.jit
def triton_arange(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(0, BLOCK)
    val = tl.arange(START, END)
    tl.store(z + off, val)

@pytest.mark.parametrize('param_list',[[0, 128],])
def test_case_access(param_list):
    start, end = param_list
    shape = [end]
    block = end - start
    dtype = 'int32'
    y_cal = torch.zeros(shape, dtype=torch.int32).npu()
    triton_arange[(1, )](y_cal, START = start, END = end, BLOCK = block)
```

