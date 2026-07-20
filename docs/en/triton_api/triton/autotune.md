# triton.autotune


```python
triton.autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=25, rep=100, use_cuda_graph=False)
```


Decorator for automatically tuning `triton.jit` functions.


```python
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size'] # the two above configs will be evaluated anytime  每当 x_size 的值发生变化时，
                 # the value of x_size changes                      上述两个配置都会被评估。
)
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE']
```


- Note: When all configurations are parsed, the kernel will run multiple times. That is, any values updated by the kernel will be updated multiple times. To avoid this undesired behavior, the `reset_to_zero` parameter can be used, which resets the provided tensor values to zero before running any configuration.
- Note: If the environment variable `TRITON_PRINT_AUTOTUNING` is set to `"1"`, Triton will print a message to stdout after each autotuning of the kernel, including the time spent on autotuning and the best configuration.


**Parameters:**


- `configs (list[triton.Config])` - A list of `triton.Config` objects.
- `key (list[str])` - A list of parameter names; when their values change, it triggers parsing of all configurations.
- `prune_configs_by (dict)` - A dictionary of functions for pruning configurations. It contains the following fields:
  - `'perf_model'`: A performance model used to predict the runtime of different configurations, returning the runtime.
  - `'top_k'`: The number of configurations to benchmark.
  - `'early_config_prune'` (optional): A function for early pruning of configurations (e.g., `num_stages`). It takes `configs: List[Config]` as input and returns the pruned configurations.
- `reset_to_zero (list[str])` - A list of parameter names that will be reset to zero before any configuration parsing.
- `restore_value (list[str])` - A list of parameter names whose values will be restored after any configuration parsing.
- `pre_hook (lambda args, reset_only)` - A function to be called before invoking the kernel. This parameter overrides the default `pre_hook` of `reset_to_zero` and `restore_value`.
  - `args`: The list of arguments passed to the kernel.
  - `reset_only`: A boolean indicating whether the `pre_hook` is only used to reset values without a corresponding `post_hook`.
- `post_hook (lambda args, exception)` - A function to be called after invoking the kernel. This parameter overrides the default `post_hook` of `restore_value`.
  - `args`: The list of arguments passed to the kernel.
  - `exception`: The exception raised by the kernel in case of compilation or runtime errors.
- `warmup (int)` - The warm-up time (in milliseconds) passed to the benchmark, default is 25.
- `rep (int)` - The repetition time (in milliseconds) passed to the benchmark, default is 100.
- `use_cuda_graph (bool)` - Whether to use CUDA Graph for performance measurement (default is `False`).

