# triton.language.parallel


## 1. Function Overview


`parallel` is an iterator specifically designed for multi-core parallel execution, inheriting from the `range` class and providing explicit multi-core parallel semantics.


```python
triton.language.parallel(arg1, arg2=None, step=None, num_stages=None,
                         loop_unroll_factor=None, bind_sub_block: bool = False,
                         _semantic=None)
```


## 2. Specifications


### 2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arg1` | `int` / `constexpr` | Required | Start value (when single parameter, used as end value, starting from 0) |
| `arg2` | `int` / `constexpr` | - | End value (not included in the range) |
| `step` | `int` / `constexpr` | `1` | Step increment per iteration |
| `num_stages` | `int` | - | Number of pipeline stages (number of iterations executed simultaneously) |
| `loop_unroll_factor` | `int` | - | Loop unroll factor (<2 means no unrolling) |
| `bind_sub_block` | `bool` | `False` | **Key parameter**: bind to sub-block, enable multi-core parallel execution |
| `_semantic` | - | - | Reserved parameter, external calls not supported for now |


> **Note**: `parallel` removes the following parameters compared to `range`:
>
> - `disallow_acc_multi_buffer`
> - `flatten`
> - `warp_specialize`
> - `disable_licm`


### 2.2 Type Support


A3:


| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | × | × | ✓ | × | × | × | × | × |


### 2.3 Special Restrictions Explanation


When `bind_sub_block` is true, the difference from `range` is reflected in the IR; whether the functionality is implemented remains to be verified.


## 3. Usage


```python
@triton.jit
def parallel_kernel(
    input_ptr,
    output_ptr0,
    output_ptr1,
    pd_ptr,
    n_elements : tl.constexpr,
):
    x = tl.arange(0,n_elements)
    x0 = x // 4
    x1 = x % 4

    a_ptr = input_ptr + x0
    b_ptr = input_ptr + x0
    for i in tl.parallel(0, 3, 1, bind_sub_block = False):
        a_ptr += x0
        b_ptr += x0
    a_ptr += x1
    b_ptr += x1
    val = tl.load(a_ptr + 0)
    tl.store(output_ptr0 + x,val)
    val = tl.load(b_ptr)
    tl.store(output_ptr1 + x,val)
```

