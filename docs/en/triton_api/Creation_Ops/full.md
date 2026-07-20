# triton.language.full


## 1. OP Overview


Introduction: `triton.language.full` returns a tensor filled with a scalar value of a given shape and data type.


```python
triton.language.full(shape, value, dtype, _semantic=None)¶
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ----------------- | ---------------------------- |
| `shape` | `tuple of ints` | Shape of the new array, e.g., (8, 16) or (8, ) |
| `value` | `scalar` | Scalar value used to fill the array |
| `dtype` | `tl.dtype` | Data type of the new array, e.g., tl.float16 |
| `_semantic` | `Optional[str]` | Reserved parameter, external calls not supported for now |


Return value:  
`tensor`: the tensor after padding is completed


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


#### 2.2.2 Shape Support


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms.


### 2.3 Special Restrictions


> Relative community capability is lacking and cannot be achieved  
> None


### 2.4 Usage


The following example implements returning a tensor of shape (XB, YB, ZB) filled with the value 100:


```python
@triton.jit
def fn_f32(output_ptr,XB : tl.constexpr,YB : tl.constexpr,ZB : tl.constexpr):
    xidx=tl.arange(0,XB)
    yidx=tl.arange(0,YB)
    zidx=tl.arange(0,ZB)

    ret = tl.full((XB,YB,ZB),value = 100,dtype = tl.float32)

    oidx=xidx[:,None,None]*YB*ZB+yidx[None,:,None]*ZB+zidx[None,None,:]

    tl.store(output_ptr+oidx,ret)
```

