# triton.language.make_tensor_descriptor


## 1. OP Overview


Introduction: Create a tensor descriptor object
Prototype (Triton 3.4.0 version):


```python
triton.language.make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    _semantic=None
) -> tensor_descriptor
```


## 2. OP Specification


### 2.1 Parameter Description


| Parameter Name | Type                | Description                                                                 |
| ------------- | ----------------- | -------------------------------------------------------------------------- |
| `base`        | `tensor`          | Base pointer of the tensor                                                 |
| `shape`       | `List[tensor]`    | Shape of the tensor                                                        |
| `strides`     | `List[tensor]`    | Stride list for each dimension of the tensor, with the following constraints: - The preceding dimensions must be integer multiples of 16 bytes - The last dimension must be contiguous in memory |
| `block_shape` | `List[constexpr]` | Shape of the block loaded from / stored to global memory                   |
| `_semantic`   | -                 | Reserved parameter; external calls are not supported for now               |


Return value:  
`tensor_descriptor`: A tensor descriptor object (cannot be directly used for arithmetic operations; must be used with `load` / `store`).


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × |
|Ascend A2/A3| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | × |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensor |
| Ascend A2/A3 | Only supports 1~5D tensor |


Conclusion: In terms of shape, there is no difference between GPU and Ascend platforms, both supporting 1 to 5-dimensional tensors.


### 2.3 Special Limitations


Relative community capability is lacking and cannot be achieved.


Conclusion: Ascend lacks support for uint16, uint32, and uint64 compared to GPU (hardware limitation).


| Difference | Description | Solution |
| ----------- | ----------- | -------- |
| Binding usage restrictions | `make_tensor_descriptor` / `load_tensor_descriptor` / `store_tensor_descriptor` must be used together and cannot be mixed with `tl.load()` / `tl.store()`. | Upgrading to Triton 3.4.0 to synchronize upstream functions (such as `cast`) can resolve this. |
| `padding_option` parameter not supported | The current community main branch has added the `padding_option` parameter for out-of-bounds element padding strategies. | Software development support is available. |
| Triton version compatibility | Triton 3.2.0 has compatibility issues with some functions (such as `cast`). It is recommended to upgrade the Triton version to 3.4.0 to fix the binding restrictions. | Upgrade to Triton 3.4.0 |


### 2.4 Usage


The following example implements an in-place absolute value calculation on the input tensor `x`:


```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    # 创建张量描述符
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
 # 计算当前线程对应的偏移量
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
 # 加载数据，计算绝对值，存储结果
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))
## 初始化张量
M, N = 256, 256
x = torch.randn(M, N, device="npu")
## 配置块大小和网格
M_BLOCK, N_BLOCK = 32, 32
grid = (M // M_BLOCK, N // N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```

