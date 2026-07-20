# triton.language.load_tensor_descriptor


## 1. OP Overview


Introduction: This function is used to load a data block from a tensor descriptor.


```python
triton.language.load_tensor_descriptor(
    desc: tensor_descriptor_base,
    offsets: Sequence[constexpr | tensor],
    _semantic=None
) -> tensor
```


## 2. OP Specification


### 2.1 Parameter Description


| Parameter | Type | Description |
| ----------- | ------------------------------- | ---------------------------------------------------------- |
| `desc` | `tensor_descriptor_base` | Tensor descriptor object, created by `make_tensor_descriptor`, defining the memory layout (shape, stride, block size, etc.). |
| `offsets` | `Sequence[constexpr \| tensor]` | Sequence of starting offsets for data loading, used to specify the data position to be loaded by the current thread block. |
| `_semantic` | - | Reserved parameter, currently not supported for external calls. |


Return value: `tensor` - The data block loaded from the specified offset based on the memory layout information of the tensor descriptor.


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
| Ascend | Only supports 1~5D tensor |


Conclusion: In terms of shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.


### 2.3 Special Limitations


Relative community capability is lacking and cannot be achieved.


Conclusion: Ascend lacks support for uint16, uint32, and uint64 compared to GPU (hardware limitation).


| Difference | Description | Solution |
| ----------- | ----------- | -------- |
| Binding usage restriction | `make_tensor_descriptor` / `load_tensor_descriptor` / `store_tensor_descriptor` must be used together and cannot be mixed with `tl.load()` / `tl.store()`. | Upgrading to Triton 3.4.0 to synchronize upstream functions (e.g., `cast`) can resolve this issue. |
| Triton version compatibility | Triton 3.2.0 has compatibility issues with some functions (e.g., `cast`). It is recommended to upgrade the Triton version to 3.4.0 to fix the binding restriction. | Upgrade to Triton 3.4.0 |


### 2.4 Usage


`load_tensor_descriptor` provides two calling forms:


* Object-oriented method invocation (recommended)


```python
value = desc.load(offsets)
```


* Functional interface call


```python
value = triton.language.load_tensor_descriptor(desc, offsets)
```


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

