# triton.Config


```python
class triton.Config(self, kwargs, num_warps, num_stages, num_ctas, maxnreg, pre_hook)
```


An object representing the kernel configurations that auto-tuning may attempt.


**Variables:**


- kwargs – A dictionary of keyword arguments to be passed to the kernel.


- num_warps – The number of threads used by the kernel when compiling for GPU. For example, if num_warps=8, each kernel instance will be automatically parallelized, using 8 * 32 = 256 threads to execute cooperatively.


- num_stages – The number of stages the compiler should use when software pipelining loops. This is useful for matrix multiplication workloads on SM80+ GPUs.


- num_ctas - Number of blocks in a block cluster. Only applicable to SM90+.


- maxnreg - The maximum number of registers that a single thread can use. Corresponds to the `.maxnreg` directive in PTX. Not supported on all platforms.


- pre_hook – A function that will be called before invoking the kernel. The parameter of this function is args.


```python
__init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None)
```


**Method:**


| init(self, kwargs[, num_warps, ...]) |
|-----|
|all_kwargs (self)|

