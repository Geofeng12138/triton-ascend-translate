# triton.language.cat


## 1. OP Overview


Introduction: The `triton.language.cat` function is used to concatenate specified tensors.


```python
triton.language.cat(input, other, can_reorder=False, _semantic=None)
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type | Description |
| ------------- | ---- | ----------- |
| `input` | `Tensor` | The first tensor to concatenate |
| `other` | `Tensor` | The second tensor to concatenate |
| `can_reorder` | `Bool` | Reorder – compiler hint. If true, the compiler is allowed to reorder elements when concatenating inputs. Only `can_reorder=True` is supported. |
| `_semantic` | `Optional[str]` | Reserved parameter; external calls are not supported at this time |


Return value:  
`tensor`: the tensor after concatenation


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


#### 2.2.2 Shape Support


Conclusion: There is no difference between GPU and Ascend platforms in terms of Shape. cat only supports concatenation of 1D shapes.


### 2.3 Special Restrictions Explanation


Relative community capability is lacking and cannot be achieved


1. Both ASCEND and CUDA only support `can_reorder=True`, which means reordering after concatenating tensors.  
2. `cat` only supports concatenation of 1D shapes.


### 2.4 Usage


The following example demonstrates concatenation of two tensors with a 1D shape:


```python
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common
import math

@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, XB: tl.constexpr):

    idx = tl.arange(0, XB)
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.cat(X, Y, can_reorder=True)

    oidx = tl.arange(0, XB * 2)

    tl.store(output_ptr + oidx, ret)


## The CAT operator in the Triton community also does not support boolean types.
@pytest.mark.parametrize('shape', [(32,), (741,)]) #triton only support 1D cat
@pytest.mark.parametrize('dtype', ['float32',])
def test_cat(shape, dtype):
    m = shape[0]
    x = torch.full((m, ), 100, dtype=eval("torch." + dtype)).npu()
    y = torch.full((m, ), 30, dtype=eval("torch." + dtype)).npu()

    output = torch.randint(1, (m * 2, ), dtype=eval("torch." + dtype)).npu()

    ans = torch.cat((x, y), dim=0)

    fn_npu_[1, 1, 1](output, x, y, m)

    test_common.validate_cmp(dtype, ans, output)
```

