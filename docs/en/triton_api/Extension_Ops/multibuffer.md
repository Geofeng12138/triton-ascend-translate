# triton.language.multibuffer


## 1. OP Overview


Introduction: Set multiple buffers for a tensor, allowing the compiler to create multiple copies of the same tensor.
Prototype:


```python
triton.language.multibuffer(
    src,
    size,
    _builder=None
) -> None
```


## 2. OP Specifications


### 2.1 Parameter Description


| Parameter Name | Type               | Description                                                      |
| ------------- | ------------------ | ---------------------------------------------------------------- |
| `src`         | `tensor`           | The source tensor for which multi-buffer setup is required       |
| `size`        | `int` or `constexpr` | Number of buffer copies to create                                |
| `_builder`    | -                  | Reserved parameter, external calls are not supported for now     |


Return value:  
`None`: This operation is a compilation hint that does not return a value at runtime and only affects the compiler's optimization behavior.


### 2.2 Supported Specifications


#### 2.2.1 DataType Support


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √      | √      | √     | √    | √    | √    | √    |


#### 2.2.2 Shape Support


Supports tensors of arbitrary shape.


### 2.3 Special Limitations


| Limit Parameter | Description |
| ----------------- | ----------- |
| `size` | The current implementation only supports `size` as `2`. |


### 2.4 Usage


The following example demonstrates how to set up multi-buffering for tensor `tmp0` in the kernel and use it in combination with other compilation hints:


```python
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        # 为 tmp0 设置双缓冲
        tl.multibuffer(tmp0, 2)
        tmp2 = tmp0
        tl.compile_hint(tmp2, "hint_b", 42)
        tl.compile_hint(tmp2, "hint_c", True)
        tl.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```

