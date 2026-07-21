# Autotune


If you want to first understand the recommended usage of Triton-Ascend autotune, the meaning of `configs=[]`, and the applicable boundaries of automatic Tiling, it is recommended to read the [Triton-Ascend autotune usage guide](../autotune_guide.md).


In this section, we will demonstrate how to use Triton's autotune method to automatically select the optimal kernel configuration parameters. The current Triton-Ascend autotune is fully compatible with the community's autotune usage (refer to the [community documentation](https://triton-lang.org/main/python-api/generated/triton.autotune.html)), meaning users need to manually pass in some predefined `triton.Config` objects, and then autotune selects the optimal kernel configuration through benchmarking. Additionally, Triton-Ascend provides an **advanced autotune** usage, where users do not need to provide information such as the kernel's split axes or tiling axes. Instead, autotune automatically parses the split axes, tiling axes, and other information based on the Triton kernel semantics, automatically generates some potentially optimal kernel configurations, and then selects the best configuration through benchmarking or profiling.


Note:  
Currently, Triton-Ascend autotune supports block size and multibuffer (compiler optimization). Due to differences in hardware architecture, the `num_warps` and `num_stages` parameters are not supported. More autotune tunable options will be continuously added in the future.


## Community autotune usage example


```Python
import torch, torch_npu
import triton
import triton.language as tl

def test_triton_autotune():

    # 返回一组不同的 kernel 配置，用于 autotune 测试
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': False}),
            triton.Config({'XS': 8 * 1024, 'multibuffer': True}),
        ]

    @triton.autotune(
        configs=get_autotune_config(),      # 配置列表
        key=["numel"],                      # 当numel大小发生变化时会触发autotune
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        XS: tl.constexpr                  # 块大小，用于控制每个线程块处理多少数据
    ):
        pid = tl.program_id(0)            # 获取当前 program 的 ID
        idx = pid * XS + tl.arange(0, XS) # 当前线程块处理的 index 范围
        msk = idx < numel                 # 避免越界的掩码

        # 重复执行一些计算以模拟负载（并测试性能）/ Repeat computation to simulate load (for perf test)
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)  # 加载 x0
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)  # 加载 x1
            tmp2 = tl.math.exp(tmp0) + tmp1 + i                # 计算
            tl.store(out_ptr0 + idx, tmp2, mask=msk)           # 存储到输出

    # Triton 调用函数，自动使用 autotuned kernel
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["XS"]), 1, 1)  # 计算 grid 大小
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # 使用 PyTorch 作为参考实现进行对比
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"                         # 使用 NPU 作为设备
    DTYPE = torch.float32
    N = 192 * 1024                      # 输入长度
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)  # 随机输入 x0
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)  # 随机输入 x1
    torch_ref = torch_calc_func(x0, x1)              # 得到参考结果
    triton_cal = triton_calc_func(x0, x1)            # 运行 Triton kernel
    torch.testing.assert_close(triton_cal, torch_ref)  # 验证输出是否一致

if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")  # 输出成功标志 / Print success message
```


## Advanced autotune usage example


```Python
# 下面说明进阶 autotune 与社区版的参数使用要点
#
# configs：
# - 社区版 autotune（默认）需要显式传入一组 triton.Config，框架会对这些配置逐一编译并基准测试以选择最优配置
# - 进阶版 autotune 框架基于 kernel 自动生成候选 tiling 配置，并对配置逐一编译并基准测试以选择最优配置
# * 注意：1. 进阶模式启动需用户手动 import triton.backends.ascend.runtime;
#        2. 若 configs=[]，框架基于 kernel 自动生成候选 tiling 配置，注意此时需要将@triton.autotune装饰器直接应用在@triton.jit之上，
#           中间不能插入其他装饰器，例如libentry;
#        3. 若 configs 不为空，则框架默认不会自动生成候选 tiling 配置;
#        4. 若 configs 不为空，且hints.auto_gen_config=True,则框架自动生成Config,并与用户定义Config合并进行配置择优；
#        5. 进阶版本支持通过设置os.environ["TRITON_BENCH_METHOD"] = ( "npu" ) 来设置性能采集方式。
#
# hints(Dict[str, str])：
# 注意：1. hints可选，用户不填时框架会自动解析切分轴（split_params），分块轴（tiling_params）等相关参数
#      2. 用户可通过hints传参来生成tiling,涉及切分轴（split_params）、分块轴（tiling_params）、低维轴（low_dim_axes）、规约轴（reduction_axes），且四个参数需同时提供

# split_params (Dict[str, str]): axis name: argument name组成的字典, argument 是切分轴的可调参数, 例如 'XBLOCK'
#     axis name必须在参数key的轴名称集合里。 请勿在轴名称前添加前缀 r
#     此参数可以为空，当split_params 和 tiling_params 都为空的时候不会进行自动寻优
#     切分轴通常可以根据 `tl.program_id()` 分核语句来确定
# tiling_params (Dict[str, str]): axis name: argument name组成的字典， argument 是分块轴的可调参数, 例如 'XBLOCK_SUB'
#     axis name必须在参数key的轴名称集合里。请勿在轴名称前添加前缀 r
#     此参数可以为空，当split_params 和 tiling_params 都为空的时候不会进行自动寻优
#     分块轴通常可以根据 `tl.arange()` 分块表达式来确定
# low_dim_axes (List[str]): 所有低维轴的轴名称列表，axis name必须在参数key的轴名称集合里
# reduction_axes (List[str]): 所有规约轴的轴名称列表，axis name必须在参数key的轴名称集合里， 在轴名称前添加前缀 r
# auto_gen_config (bool): 默认为False,涉及如下场景组合
#     1. 用户未定义Config,无论是否设置auto_gen_config,框架默认自动生成Config；
#     2. 用户定义了Config,且auto_gen_config=False,则框架不自动生成Config,只使用用户定义的Config；
#     3. 用户定义了Config,且auto_gen_config=True,则框架自动生成Config,并与用户定义Config合并进行配置择优；
#
# key（list[str]/Dict[str,str]）：
# - 传入运行时参数名列表；列表中任一参数值变化会触发候选配置的重新生成与评估
# 注意：1.若hints传递切分轴（split_params）、分块轴（tiling_params）、低维轴（low_dim_axes）、规约轴（reduction_axes）参数信息，key类型需为Dict[str,str],如示例1：
#      2.若hints不传递切分轴（split_params）、分块轴（tiling_params）、低维轴（low_dim_axes）、规约轴（reduction_axes）参数信息，key类型需为list[str]，轴信息会按参数顺利进行分配，如示例2：

示例1:
@triton.autotune(
    configs=[],
    key={"x":"n_elements"},
    hints={
        "split_params":{"x":"BLOCK_SIZE"},
        "tiling_params":{},
        "low_dim_axes":["x"],
        "reduction_axes":[],
    }
)
示例2:
@triton.autotune(
    configs=[],
    key=["n_elements"],
)
@triton.jit
def add_kernel(
    x_ptr,  # *指向*第一个输入向量的指针。
    y_ptr,  # *指向*第二个输入向量的指针。
    output_ptr,  # *指向*输出向量的指针。
    n_elements,  # 向量的大小。
    BLOCK_SIZE: tl.constexpr,  # 每个核应该处理的元素数量。
    # 注意：`constexpr` 表示它可以在编译时确定，因此可以作为形状（shape）值使用。
):
    pid = tl.program_id(axis=0)  # 我们使用一维的grid，因此轴为0。
    # 当前核将处理的数据在内存中相对于起始地址的偏移。
    # 例如，如果你有一个长度为256的向量，且块大小（block_size）为64，那么各个程序
    # 将分别访问元素 [0:64, 64:128, 128:192, 192:256]。
    # 注意，offsets 是一个指针列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建一个掩码（mask），以防止内存操作访问越界。
    mask = offsets < n_elements
    # 加载x和y，并使用掩码屏蔽掉多余的元素，以防输入向量的长度不是块大小的整数倍。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将 x + y 写回。
    tl.store(output_ptr + offsets, output, mask=mask)
```


Note:


1. By default, Triton-Ascend uses a benchmark method to obtain on-chip computation time. When the environment variable `export TRITON_BENCH_METHOD="npu"` is set, it uses `torch_npu.profiler.profile` to obtain the on-chip computation time for each kernel configuration. For some triton kernels with fast computation, such as small-shape operators, this method can obtain more accurate computation time compared to the default method. However, it will significantly increase the overall autotune time, so please enable it with caution.  
2. Currently, this advanced usage is only applicable to Vector-type operators and does not support Cube-type operators. For more advanced usage examples, please refer to [autotune advanced usage examples](https://gitcode.com/Ascend/triton-ascend/tree/main/third_party/ascend/unittest/autotune_ut/).


### Automatic Parameter Parsing


Before performing automatic parameter parsing, it first retrieves the parameters that were not passed when calling the `kernel` function, **using the unpassed parameters as candidates for the split axis and block axis parameters**.


```Python
@triton.jit
def kernel_func(
    outputptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    # kernel implementation
    ...

# XBLOCK和XBLOCK_SUB未传入，则作为切分轴和分块轴参数的候选项
# BLOCK_SIZE以关键字参数传入，不作为参数候选项，不会被识别
kernel_func[grid](y, x, n_rows, n_cols, BLOCK_SIZE=block_size)
```


#### Shard Axis Parameter Parsing


The split axis parameter parsing is determined based on the `tl.program_id()` kernel dispatch statement. The system identifies potential split axis parameters by analyzing the usage of the `tl.program_id()` variable in the program and its multiplication operations with other variables (currently supporting scenarios of direct multiplication or indirect multiplication through intermediate variables), and filters them according to the candidate parameter list (parameters not provided by the user).


Finally, confirm the split axis corresponding to the current parameter through mask comparison and the `key` passed in `autotune`.


Note: 1. The split axis parameter must be multiplied by `tl.program_id()`. 2. A mask comparison must be performed, and the key corresponding to this axis must be directly used as the right-hand value or as the right-hand value of a min function with the key as its parameter, in order to correspond to the specific split axis; otherwise, parameter parsing will fail. 3. The identified split axis parameters are limited to the candidate parameter list, ensuring that only those parameters that can be dynamically adjusted through auto-tuning are considered.


```Python
@triton.autotune(
    configs=[],
    key={"n_elements"} # 需要指定
    ...
)
@triton.jit
def triton_func(...):
    # case1:
    pid = tl.program_id(0)
    block_start = pid * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)

    # case2:
    block_start = tl.program_id(0) * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)

    # case3:
    offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)

    # mask compare
    mask = offsets < n_elements # 1
    mask = offsets < min(..., n_elements) # 2

# 解析得到切分轴参数 split_params = {"x": "XBLOCK"}
```


#### Block Axis Parameter Parsing


Block axis parameters are determined based on block statements such as `tl.arange()`, `tl.range()`, and `range()`. By analyzing the usage of `tl.range()`, `tl.arange()`, and `range()` in `for` loops within the program, along with the variables computed from them, potential block axis parameters are identified. Common parameters between `tl.range()` or `range()` and `tl.arange()` are extracted and filtered according to the candidate parameter list (parameters not provided by the user).


Finally, confirm the tiling axis corresponding to the current parameter through mask comparison and the `key` passed in `autotune`.


Note: 1. The block axis parameter must appear in the call to `tl.arange()` and participate in the calculation of the loop range within the `for` loop via `tl.range()`, `range()`, or integer division (`//`). 2. A mask comparison must be performed, and the key corresponding to that axis must be used directly as the right-hand value or as the right-hand value via a `min` function with the key as its parameter, in order to map to the specific block axis; otherwise, parameter parsing will fail. 3. The identified block axis parameters are limited to the candidate parameter list, ensuring that only those parameters that can be dynamically adjusted through autotuning are considered.


```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # 需要指定
    ...
)
@triton.jit
def triton_func(...):
    ...
    # case 1
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

    # case 2
    loops = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop in range(loops):
        row_offsets = loop * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

        ...
        xmask = row_offsets < n_rows # 1
        xmask = row_offsets < min(..., n_rows) # 2
        ymask = col_offsets < n_cols

# 解析得到分块轴参数 tiling_params = {"x": "XBLOCK_SUB"}
# 参数BLOCK_SIZE虽然也在tl.arange中且与n_cols比较计算mask，但不是一个分块轴参数
```


#### Low-Dimensional Axis Parameter Parsing


Low-dimensional axis parameter parsing is determined based on the `tl.arange()` block statement. By analyzing the usage of `tl.arange()` in the program and the variables it computes, potential low-dimensional axis parameters are identified. The `tl.arange()` itself and the variables involved in its computation are extracted. Dimensionality is increased based on whether slicing operations are performed, and filtering is done by determining the increased dimension.


Finally, confirm the low-dimensional axis of the current kernel through mask comparison and the `key` passed into `autotune`.


Note: 1. The low-dimensional axis must be computed using `tl.arange()` and sliced. It will only be recognized if it undergoes dimension expansion on non-lowest dimensions or does not participate in slicing. 2. Without mask comparison, it cannot correspond to a specific low-dimensional axis, which will cause parameter parsing to fail.


```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # 会按顺序自动分配成 {"x": "n_rows", "y": "n_cols"}
    ...
)
@triton.jit
def triton_func(...):
    ...
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols

# 解析得到低维轴 low_dim_axes = {"y"}
# row_offsets虽然也通过tl.arange计算且与n_rows比较计算mask，但切片在低维进行扩充，所以x不是一个低维轴
```


#### Parameter Pointer Parsing


Pointer-type parameter resolution is determined based on whether the parameter participates in memory access statements such as `tl.load()` and `tl.store()`.


First, parse all parameters in the kernel function, then recursively find all variables involved in the computation for each parameter.


If this parameter directly participates in, or the intermediate variable calculated from this parameter indirectly participates in, the computation of the first argument of `tl.load()` and `tl.store()`, then this parameter is considered a pointer type parameter.


Note: 1. Variables decorated with `tl.constexpr` will not be pointer-type variables and will not undergo subsequent parsing. 2. Only memory access statements where parameters directly participate or intermediate variables obtained through a single computation of parameters indirectly participate are calculated. Intermediate variables obtained through two or more computations of parameters are not counted.


```Python
@triton.autotune(...)
@triton.jit
def triton_func(input_ptr, output_ptr, ...):
    ...
    # case1
    input = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input, mask=mask)

    # case2
    inputs_ptr = input_ptr + offsets
    input = tl.load(inputs_ptr, mask=mask)
    outputs_ptr = output_ptr + offsets
    tl.store(outputs_ptr, input, mask=mask)

# 解析得到指针类型参数为：input_ptr, output_ptr
```


## More Features


### Profiling Results for Automatically Generating Optimal Configuration


```Python
# 自动在`auto_profile_dir`目录中生成当前autotune最优kernel配置的profiling结果，即利用`torch_npu.profiler.profile`采集的性能数据
# 在社区autotune用法和进阶autotune用法中均可生效
@triton.autotune(
    auto_profile_dir="./profile_result",
    ...
)
```

