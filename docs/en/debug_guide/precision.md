# Precision Comparison and Error Analysis


This document describes how to perform precision comparison and error analysis of Triton operators on the Ascend NPU, including comparison methods, evaluation criteria, and precautions.


## 1. Precision Comparison Process


### Basic Steps


1. **Obtain Reference Results** (Golden): Use equivalent Torch operators to compute results on CPU/GPU/NPU, or use the same Triton operator to compute results on CPU/GPU.


2. **Obtain Triton Results**: Run the Triton kernel on the Ascend NPU to obtain the computation results.


3. **Comparison Judgment**: Use `torch.testing.assert_close` to determine whether the precision requirements are met.


### Example: Vector Add


```python
import torch
import triton
import triton.language as tl


def test_vector_add(n, dtype):
    # 1. 输入数据
    x = torch.randn(n, dtype=dtype, device="cpu")
    y = torch.randn(n, dtype=dtype, device="cpu")

    # 2. 参考结果（PyTorch CPU）
    torch_ref = x + y

    # 3. Triton kernel
    @triton.jit
    def add_kernel(in0_ptr, in1_ptr, out_ptr, n: tl.constexpr):
        idx = tl.arange(0, n)
        a = tl.load(in0_ptr + idx)
        b = tl.load(in1_ptr + idx)
        tl.store(out_ptr + idx, a + b)

    def triton_func(x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x.npu(), y.npu(), out, n=x.numel())
        return out

    triton_cal = triton_func(x, y)

    # 4. 精度对比
    compare_precision(triton_cal.cpu(), torch_ref)
```


## 2. Precision Comparison Function


```python
def compare_precision(cal, ref):
    """
    精度对比函数：根据数据类型选择合适的比对策略。

    参数:
        cal: 计算结果
        ref: 参考结果
        rtol: 相对误差容限
        atol: 绝对误差容限

    异常:
        AssertionError: 精度不达标时抛出
    """
    assert cal.dtype == ref.dtype, f"dtype mismatch: {cal.dtype} vs {ref.dtype}"
    tensor_dtype = cal.dtype

    if tensor_dtype == torch.float16:
        torch.testing.assert_close(ref, cal, rtol=1e-3, atol=1e-3, equal_nan=True)

    elif tensor_dtype == torch.bfloat16:
        torch.testing.assert_close(ref, cal, rtol=5e-3, atol=5e-3, equal_nan=True)

    elif tensor_dtype == torch.float32:
        torch.testing.assert_close(ref, cal, rtol=1e-5, atol=1e-5, equal_nan=True)

    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        assert torch.equal(cal, ref), f"Integer tensors are not equal for dtype {tensor_dtype}"

    elif tensor_dtype == torch.bool:
        assert torch.equal(cal, ref), "Boolean tensors are not equal"

    else:
        raise ValueError(f"Unsupported tensor dtype: {tensor_dtype}")

    print(f"dtype: {tensor_dtype} — Precision check passed.")
```


## 3. Precision Judgment Criteria


### Judgment Rules


`torch.testing.assert_close`/`torch.equal` does not raise an exception → **Pass**, otherwise **Fail**.


* `torch.testing.assert_close`: Passes (does not raise an exception) if tensors are approximately equal within the specified tolerance; otherwise fails (raises AssertionError).


* `torch.equal`: Returns True only if two tensors have exactly the same shape and all elements are absolutely equal at the binary level; otherwise, returns False.


The internal logic of `torch.testing.assert_close`:


```
|cal - ref| <= atol + rtol * |ref|
```


That is: the absolute error `|cal - ref|` must satisfy a dynamic error boundary formed by the sum of the relative error tolerance and the absolute error tolerance.


### Recommended Tolerances by Data Type


| Data Type | rtol | atol | Description |
|---|---|---|---|---|
| `float32` | 1e-5 | 1e-5 | Strict |
| `float16` | 1e-3 | 1e-3 | Low precision, appropriately relaxed |
| `bfloat16` | 5e-3 | 5e-3 | Low precision, appropriately relaxed |
| `int8/16/32/64` | — | — | Must be exactly identical (`torch.equal`) |
| `bool` | — | — | Must be exactly identical (`torch.equal`) |



## 4. Precautions


### NaN / Inf Handling


`equal_nan=True` treats NaN as equal. If strict detection of NaN differences is required, set it to `False`.


### Integer Types


Integer and boolean types do not allow any deviation and must be strictly identical. When comparing across devices, ensure that the data has been moved to the same device (e.g., CPU) to avoid misjudgment caused by differences in underlying representation.

