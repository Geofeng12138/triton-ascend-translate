# triton.heuristics


```python
triton.heuristics(values)
```


Decorator used to specify how to calculate certain meta-parameter values. This is particularly useful in scenarios where automatic tuning is too costly or not applicable.


```python
@triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size  最小的 2 的幂 >= x_size
```


**Parameters:** `values (dict[str, Callable[[list[Any]], Any]]**)` - A dictionary containing meta-parameter names and functions that compute meta-parameter values. Each such function accepts a list of positional arguments as input.

