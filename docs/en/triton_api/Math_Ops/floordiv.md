# floordiv


## 1. OP Overview


Introduction: Floor division, returns the result of division rounded towards zero, arithmetic operator '//', no tl.floordiv method.


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type                | Description                                                  |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `self`        | `tensor or Number`     |     First input, dividend    |
| `other`       | `tensor or Number`     |     Second input, divisor    |


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √| √    | ×    | ×    | ×    | √    |
|Ascend A2/A3| × | √ | × | √ | × | √ | × | √  | ×    | ×    | ×    | √    |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Unlimited               |
| Ascend A2/A3 | Unlimited               |


Conclusion: There is no difference between GPU and Ascend platforms in terms of Shape.


### 2.3 Usage


The following example performs floor division on the input tensors `in_ptr0, in_ptr1`:


```python
@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx)
    y = tl.load(in_ptr1 + idx)
    ret = x // y
    tl.store(out_ptr0 + idx, ret)
```


### 2.4. Special Restrictions


Ascend A3 lacks support for uint8, uint16, uint32, and uint64 compared to GPU.

