# triton.language.sync_block_set


## 1. Function Overview


Explicit inter-core synchronization instruction used to coordinate execution order and data consistency among different cores in the Cube-Vector architecture.


## 2. `sync_block_set` Operation


### 2.1 Function Overview


After the producer core completes its task, it sends a synchronization signal to the consumer core.


```python
triton.language.sync_block_set(sender, receiver, event_id, _builder=None)
```


### 2.2 Specifications


#### 2.2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sender` | `str` | Required | Sender core type: "cube" or "vector" |
| `receiver` | `str` | Required | Receiver core type: "cube" or "vector" |
| `event_id` | `int` | Required | Event ID, used to distinguish different synchronization points |
| `_builder` | - | `None` | Reserved parameter, external invocation not supported for now |


#### 2.2.2 Special Restrictions Explanation


1. `sender` and `receiver` cannot be the same; you cannot send a signal to yourself.
2. `event_id` must be in the range of 0-15 (16 independent events in total).


## 3. `sync_block_wait` Operation


### 3.1 Function Overview


The consumer core waits for the synchronization signal from the producer.


```python
triton.language.sync_block_wait(sender, receiver, event_id, _builder=None)
```


### 3.2 Specifications


#### 3.2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sender` | `str` | Required | Sender core type: `"cube"` or `"vector"` |
| `receiver` | `str` | Required | Receiver core type: `"cube"` or `"vector"` |
| `event_id` | `int` | Required | Event ID to wait for |
| `_builder` | - | `None` | Reserved parameter, external calls not supported yet |


#### 3.2.2 Special Limitations


1. `sender` and `receiver` must not be the same
2. `event_id` must match the ID used by the corresponding `sync_block_set`


## 4. `sync_block_all` Operation


### 4.1 Function Overview


Global barrier synchronization, synchronizing all cores of the specified type to the same point.


```python
triton.language.sync_block_all(mode, event_id, _builder=None)
```


### 4.2 Specifications


#### 4.2.1 Parameter Description


| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | Required | Synchronization mode: "all_cube", "all_vector", or "all" |
| `event_id` | `int` | Required | Global synchronization event ID |
| `_builder` | - | `None` | Reserved parameter, external invocation not supported for now |


#### 4.2.2 Special Restrictions Explanation


1. `mode` must be one of "all_cube", "all_vector", or "all"
2. `event_id` must be in the range 0-15


## 5. Usage


### 5.1 Basic Usage Example


```python
import triton
import triton.language as tl
import triton.language.ascend as al

@triton.jit
def sync_example():
    # Cube 核心计算并通知 Vector
    with al.Scope(core_mode="cube"):
        # ... 执行 Cube 计算 ...
        tl.sync_block_set("cube", "vector", 0)

    # Vector 核心等待 Cube 完成
    with al.Scope(core_mode="vector"):
        tl.sync_block_wait("cube", "vector", 0)
        # ... 执行 Vector 计算 ...
```


### 5.2 Flash Attention Pipeline Example


```python
@triton.jit
def flash_attention_fwd(q_ptr, k_ptr, v_ptr, o_ptr, ...):
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    with al.Scope(core_mode="cube"):
        for start_n in range(0, N, BLOCK_N):
            qk = tl.dot(q, k)
            tl.sync_block_set("cube", "vector", 0)
            tl.sync_block_wait("vector", "cube", 1)
            pv = tl.dot(p, v)
            tl.sync_block_set("cube", "vector", 2)

    with al.Scope(core_mode="vector"):
        for start_n in range(0, N, BLOCK_N):
            tl.sync_block_wait("cube", "vector", 0)
            m_new, l_new, softmax_out = _softmax(qk, m_prev, l_prev)
            tl.sync_block_set("vector", "cube", 1)
            tl.sync_block_wait("cube", "vector", 2)
            acc = _update_output(pv, softmax_out, acc)

    with al.Scope(core_mode="cube"):
        tl.sync_block_all("all", 0)

    tl.store(o_ptr + offsets, acc)
```

