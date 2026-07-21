# Triton-Ascend FAQ


## 1. Installation and Environment Configuration


**Q: How to correctly install Triton-Ascend? Does it support direct installation via pip?**


A: You can directly install it using pip.


```Python
pip install triton-ascend
```


**Q: Can the community Triton and Triton-Ascend coexist?**


A: For triton-ascend 3.2.0 and below, it is not possible. You need to uninstall the community Triton first, then install Triton-Ascend.<br>
For triton-ascend 3.2.1 and above, Triton-Ascend mitigates the installation overwrite issue by declaring Triton as an installation dependency.
When installing Triton-Ascend, the community Triton is installed first, and then Triton-Ascend overwrites the same-named directory, thereby preventing subsequent installations of other software packages that depend on Triton from overwriting Triton-Ascend.
The reason why x86 and arm use different versions of the community Triton installation package is that the community only provides arm version packages starting from version 3.5: x86 depends on triton==3.2.0, arm depends on triton==3.5.0.


- Note: If you install a third-party component that depends on triton, or triton itself, after installing triton-ascend, it will overwrite the installed Triton-Ascend directory. In this case, you need to uninstall the community Triton and Triton-Ascend first, and then install Triton-Ascend.


```Python
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```


**Q: Can Triton-Ascend be used on non-Ascend hardware (such as CUDA AMD)?**


A: No, it can only be used in the Ascend NPU hardware environment with Triton-Ascend.


## 2. Precision and Numerical Consistency Issues


**Q: The NPU runtime results are inconsistent with the PyTorch/CPU/GPU reference results. How to troubleshoot?**


A: For use cases, please refer to [07_accuracy_comparison_example.md](../zh/examples/07_accuracy_comparison_example.md). For debugging methods, please refer to [Interpreter Mode Debugging Methods](./debug_guide/debugging.md#5-调试方法).


## 3. Error Codes and Exception Handling


**Q: Why does kernel compilation report MLIRCompilationError? How to locate the specific failing Pass?**


A: Please refer to [Compilation Error Debugging Method](./debug_guide/debugging.md#52-编译错误调试方法)


## 4. Debugging and Logging


**Q: How to enable detailed log output? Where is TRITON_DEBUG=1 output?**


A: You can use `TRITON_DEBUG=1` to obtain detailed debug dump files. Please refer to [Debug Dump Files](./debug_guide/debugging.md#32-调试转储文件dump-files).


**Q: Can I print intermediate tensor values in the kernel? Is tl.device_print available?**


A: You can use `tl.device_print` to print tensors in the kernel. Please refer to [Print Debugging Method](debug_guide/debugging.md#51-打印调试方法).


## 5. Development and Contribution


**Q: How to build and test Triton-Ascend locally?**


A: For local build and test methods, please refer to [Install Triton-Ascend from Source](./installation_guide.md#源码编译安装)


**Q: What CI checks must be passed to submit a PR?**


A: The CI checks for a PR include: coding security and standard checks, open-source snippet checks, malicious code checks, compilation and build, and developer testing.


## 6. Performance Tuning


**Q: Is there a performance analysis tool (profiler) available?**


A: There is an integrated performance analysis tool (profiler). Please refer to [Operator Performance Tuning Method](./debug_guide/profiling.md).


## 7. UB Overflow FAQ


**Q: Compilation reports "UB Overflow" error, how to resolve?**


A: UB Overflow is a common issue during Triton-Ascend development. Please refer to the [UB Overflow Troubleshooting Guide](./debug_guide/ub_overflow.md) for problem diagnosis. If you are unsure how to reduce tiling to decrease UB usage, you can use Autotune to automatically select the optimal configuration. For instructions on using Autotune, please refer to the [Triton-Ascend Autotune Usage Guide](./autotune_guide.md).  
Operators that run on the A5 may cause UB Overflow when migrated to A2/A3 due to differences in UB size. If manual troubleshooting fails, Autotune can also be used to automatically select the optimal configuration.


## 8. Triton Usage Limitations


**Q: What are the usage restrictions for pointer parameters in Triton Kernel?**


A: Triton-Ascend assumes during compilation that all externally input pointer parameters essentially point to different memory regions and cannot recognize pointer alias scenarios. When multiple pointer parameters actually point to the same memory block at runtime, but the compiler cannot know this fact, it may lead to optimization failures or abnormal runtime results. For example:


```Python
@triton.jit
def func(ptr0, ptr1):
    # load from ptr0 and do something
    # store to ptr0
    # load from ptr1 and do something
    # store to ptr1

in_out_tensor = torch.randn(shape)
func[grid](in_out_tensor, in_out_tensor)
```


In the above code, `ptr0` and `ptr1` actually point to the same memory block (i.e., the same `in_out_tensor`), but the compiler cannot recognize this pointer aliasing relationship. Therefore, passing the same tensor as multiple pointer arguments is not supported, and the corresponding Kernel will not be able to enable related optimizations.


**Q: What are the limitations of using `tl.load` / `tl.store` in control flow OPs such as `if` / `for` / `while`?**


A: Triton-Ascend supports memory access after simple address updates of pointers from the same source within control flow. Placing `tl.load` / `tl.store` inside control flow is also a valid practice. However, it is not recommended to merge pointers from different sources or with different structures after control flow and then perform unified memory access; nor is it advisable to repeatedly update pointer states and simultaneously execute store/read-after-write operations within complex nested control flow.


The current version does not fully support scenarios where `if` / `for` / `while` are used in combination with `tl.load` / `tl.store`. This will be continuously improved in future versions. For now, it is recommended to follow the restrictions below.


It is not recommended to merge pointers with different base addresses, or block pointers constructed in different branches, after branching and then perform memory access.


```Python
if cond:
    ptr = x + offsets
else:
    ptr = y + offsets
value = tl.load(ptr)
```


It is recommended to store the visits in their respective branches, allowing the branch to merge the loaded value instead of a pointer or block pointer.


```Python
if cond:
    value = tl.load(x + offsets)
else:
    value = tl.load(y + offsets)
```

