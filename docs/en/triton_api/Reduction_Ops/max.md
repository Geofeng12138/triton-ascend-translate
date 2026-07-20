# triton.language.max


## 1. OP Overview


Introduction: Returns the maximum value along the specified dimension.


```python
triton.language.max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter | Type | Description |
| :---: | :---: | --- |
| `input` | `tensor` | The input tensor data |
| `axis` | `int` | Specifies the dimension along which to reduce; when `axis` is none, reduce over all axes |
| `keep_dims` | `bool` | Whether to keep the reduced dimensions after reduction |
| `return_indices` | `bool` | If true, return the index corresponding to the maximum value; in addition to returning the maximum value, also return the index of the maximum value |
| `return_indices_tie_break_left` | `bool` | If multiple elements have the same maximum value, return the index of the leftmost maximum value; if true, in case of a tie (i.e., multiple elements have the same maximum value), return the left-most index for values that aren’t NaN |


Return value:  
`tl.tensor`: A tensor with the same shape as `input`.  
When `return_indices = true`, the returned index type is fp32.


Parameter combination support:


| axis | keep_dims | return_indices | return_indices_tie_break_left | Specification |
|------|-----------|----------------|--------------------------------|--------------|
| 1    | TRUE      | TRUE           | TRUE                           | Supported    |
| 1    | TRUE      | TRUE           | FALSE                          | Supported    |
| 1    | TRUE      | FALSE          | TRUE                           | Supported    |
| 1    | TRUE      | FALSE          | FALSE                          | Supported    |
| 1    | FALSE     | TRUE           | TRUE                           | Supported    |
| 1    | FALSE     | TRUE           | FALSE                          | Supported    |
| 1    | FALSE     | FALSE          | TRUE                           | Supported    |
| 1    | FALSE     | FALSE          | FALSE                          | Supported    |
| None | TRUE      | TRUE           | TRUE                           | Not Supported|
| None | TRUE      | TRUE           | FALSE                          | Not Supported|
| None | TRUE      | FALSE          | TRUE                           | Supported    |
| None | TRUE      | FALSE          | FALSE                          | Supported    |
| None | FALSE     | TRUE           | TRUE                           | Not Supported|
| None | FALSE     | TRUE           | FALSE                          | Not Supported|
| None | FALSE     | FALSE          | TRUE                           | Supported    |
| None | FALSE     | FALSE          | FALSE                          | Supported    |


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | √ |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| -------- | ---------------------- |
| GPU    | No limit |
| Ascend A2/A3| No limit (default max 8 dimensions) |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms.


### 2.3 Usage


For more examples, refer to the triton-ascend repository, ascend/examples/generalization_cases/test_max.py


```@triton.jit
def triton_max_1d(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + xoffset, None)
    tmp4 = tl.max(tmp0, 0)
    tl.store(out_ptr1, tmp4, None)
```


### 2.4 Special Restrictions


Ascend A3 compared to GPU does not support uint16, uint32, uint64, fp64

