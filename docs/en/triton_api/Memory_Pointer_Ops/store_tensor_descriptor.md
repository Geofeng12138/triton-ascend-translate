# triton.language.store_tensor_descriptor


## 1. OP Overview


Introduction: Store the data block to the memory location specified by the tensor descriptor.


```python
triton.language.store_tensor_descriptor(
    desc: tensor_descriptor_base,
    offsets: Sequence[constexpr | tensor],
    value: tensor,
    _semantic=None
) -> tensor
```


## 2. OP Specification


### 2.1 Parameter Description


| Parameter | Type | Description |
| ----------- | ------------------------------- | ---------------------------------------------------------- |
| `desc`      | `tensor_descriptor_base`        | Tensor descriptor object, created by `make_tensor_descriptor`, defining the memory layout (shape, strides, block sizes, etc.). |
| `offsets`   | `Sequence[constexpr \| tensor]` | Sequence of starting offsets for data storage, used to specify the data location to be stored by the current thread block. |
| `value`     | `tensor`                        | The tensor data block to be written. |
| `_semantic` | -                               | Reserved parameter, external calls are not supported for now. |


Return value: `tensor` - the actual written data block


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × |
|Ascend A2/A3| √ | √ | × | √ | × | √ | × | √ | √ | √ | √ | × |


#### 2.2.2 Shape Support


|        | Supported Dimension Range |
| ------ | ----------------------- |
| GPU    | Only supports 1~5D tensors |
| Ascend | Only supports 1~5D tensors |


Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms, both support 1 to 5 dimensional tensors.


### 2.3 Special Limitations


Relative community capability is lacking and cannot be achieved.


Conclusion: Ascend lacks support for uint16, uint32, and uint64 compared to GPU (hardware limitation).


### 2.4 Usage


`store_tensor_descriptor` provides two calling forms:


* Object-oriented method invocation (recommended)


```python
desc.store(offsets, value)
```


* Functional interface invocation


```python
triton.language.store_tensor_descriptor(desc, offsets, value)
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

