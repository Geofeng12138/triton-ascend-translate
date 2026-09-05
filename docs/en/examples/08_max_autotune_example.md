# max_autotune Usage Example


`max_autotune` is an enhanced auto-tuning decorator provided by Triton-Ascend, designed to simplify the code writing for multi-parameter tuning. Unlike the community version `autotune`, which requires users to manually enumerate all `triton.Config` options, `max_autotune` allows users to provide only a few basic configurations (such as block sizes) and automatically incorporates related compiler options (like `num_stages`, `enable_hivm_auto_cv_balance`, etc.) into the search space for optimal combinations. Users can also explicitly control the search scope through parameter lists.


**Applicable Scenarios**: `cube`, `mix`, `vector` operators on Ascend NPU, especially suitable for scenarios that require simultaneous adjustment of multiple hardware-related parameters.


## Basic Usage Example


The following example demonstrates the use of `max_autotune` for automatic tuning of a simple vector addition kernel. Compared to the community version of `autotune`, `max_autotune` also automatically includes different compiler options in the tuning space, eliminating the need for manual specification by the user.


```Python
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.runtime import max_autotune

def test_max_autotune():

    # 基础配置：只需提供分块大小，其他调优参数由装饰器自动生成
    base_configs = [
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ]

    @max_autotune(
        configs=base_configs,
        key=["numel"],
        kernel_type="vector",          # 算子类型：cube / mix / vector, 默认为mix
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < numel

        # 模拟计算负载
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=mask, other=0.0)
            tmp1 = tl.load(in_ptr1 + idx, mask=mask, other=0.0)
            tmp2 = tl.math.exp(tmp0) + tmp1 + i
            tl.store(out_ptr0 + idx, tmp2, mask=mask)

    # 封装调用函数
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # 与 PyTorch 参考结果对比
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"
    DTYPE = torch.float32
    N = 192 * 1024
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)
    torch_ref = torch_calc_func(x0, x1)
    triton_cal = triton_calc_func(x0, x1)
    torch.testing.assert_close(triton_cal, torch_ref)

if __name__ == "__main__":
    test_max_autotune()
    print("success: test_max_autotune")
```


## Advanced Usage: Precisely Controlling Tuning Parameters


Users can explicitly specify the compiler options to be tuned and their value lists via **tuning_params**; parameters not specified will use built-in default values. The following example demonstrates how to perform a combined search for multiple parameters.


```python
from triton.backends.ascend.runtime import max_autotune

def test_max_autotune():

    # 基础配置：只需提供分块大小，其他调优参数由装饰器自动生成
    base_configs = [
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ]

    @max_autotune(
        configs=base_configs,          # 基础配置列表
        key=["numel"],                 # 当 numel 变化时触发重新调优
        kernel_type="vector",          # 算子类型：cube / mix / vector
        # 以下参数为可选的调优列表，不提供时使用内置默认值
        num_stages=[1, 2],
        enable_ubuf_saving=[True, False]
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        BLOCK_SIZE: tl.constexpr,
        **META
    ):
        pass
```

