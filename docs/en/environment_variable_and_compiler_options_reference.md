# Environment Variables and Compiler Options


This document summarizes the behavior switches in Triton-Ascend that can be explicitly controlled by developers, including environment variables set before runtime, as well as NPU compilation options passed during compilation via `triton.Config` or kernel launch meta-parameters.


## Environment Variables


### Environment Variable Usage Examples


Environment variables need to be set before running a Python program, for example:


```bash
export TRITON_DEBUG=1
python run_kernel.py
```


### Environment Variable Reference Table


Environment Variable Configuration Reference Table:


| Category | Environment Variable | Default Value | Description | Configuration Notes | Change Notes |
|------|----------|--------|----------|----------|----------|
| **Debugging & Logging** | TRITON_DEBUG | 0 or unset | Enables Triton's debug output for printing detailed debugging information at runtime. Useful for troubleshooting compilation or execution issues. When set to 1, Triton outputs more information about the compilation process, kernel generation, and execution. Some implementations may support finer-grained debug levels (e.g., 2, 3, etc.), depending on the Triton version and implementation. | 0: Do not enable DEBUG<br>1: Enable DEBUG | |
| **Debugging & Logging** | MLIR_ENABLE_DUMP | 0 or unset | Dumps the IR of all kernels before each MLIR optimization pass. Use `MLIR_ENABLE_DUMP=kernelName` to dump the IR of a specific kernel only. | 0: Do not dump<br>1: Dump IR of all kernels<br>kernelName: Dump IR of a specific kernel | Triton cache may interfere with dumping. If `MLIR_ENABLE_DUMP=1` does not take effect, try clearing the Triton cache: `rm -r ~/.triton/cache` |
| **Debugging & Logging** | LLVM_IR_ENABLE_DUMP | 0 or unset | Dumps IR before each LLVM IR optimization pass. | 0: Do not dump<br>1: Dump IR | |
| **Debugging & Logging** | TRITON_REPRODUCER_PATH | unset | Generates an MLIR reproducer file before each MLIR compilation stage. If a stage fails, `<reproducer_path>` will save the MLIR state before the failure. | <reproducer_path>: Save path | |
| **Debugging & Logging** | TRITON_INTERPRET | 0 or unset | Uses the Triton interpreter instead of running on the GPU, allowing Python breakpoints to be inserted in kernel code. | 0: Breakpoints not supported<br>1: Breakpoints supported | |
| **Debugging & Logging** | TRITON_ENABLE_LLVM_DEBUG | 0 or unset | Passes the `-debug` argument to LLVM, outputting extensive debugging information. If the output is too verbose, use `TRITON_LLVM_DEBUG_ONLY` to limit the scope. | 0: Do not pass<br>1: Pass | Another way to reduce output noise: first run the program with `LLVM_IR_ENABLE_DUMP=1`, extract the intermediate representation (IR) before the target LLVM optimization pass, then run LLVM's `opt` tool separately, adding the `-debug-only=foo` argument via the command line to limit the debug scope. |
| **Debugging & Logging** | TRITON_LLVM_DEBUG_ONLY | unset | Functions equivalently to LLVM's `-debug-only` command-line option. This parameter restricts LLVM debug output to specific optimization passes or component names (defined by `#define DEBUG_TYPE` macros in LLVM and Triton), effectively reducing redundant debug information. Users can specify one or more comma-separated values, e.g., `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`. | Comma-separated values: Pass or component names | |
| **Debugging & Logging** | USE_IR_LOC | 0 or unset | Controls whether location information (e.g., file name, line number) is included in the generated intermediate representation (IR). This information is helpful for debugging but may increase the size of the generated IR. When set to 1, the IR is re-parsed, mapping location information to line numbers in IR files with specific extensions (rather than Python source file line numbers). This establishes a direct mapping from IR to LLVM IR/PTX. When used with profiling tools, it enables fine-grained performance analysis of IR instructions. | 0: Do not include location info<br>1: Include location info | |
| **Debugging & Logging** | TRITON_PRINT_AUTOTUNING | 0 or unset | After autotuning completes, outputs the best configuration and total time for each kernel. | 0: Do not output<br>1: Output | |
| **Debugging & Logging** | MLIR_ENABLE_REMARK | 0 or unset | Enables remark output during MLIR compilation, including performance warnings output as remarks. | 0: Do not enable<br>1: Enable | |
| **Debugging & Logging** | TRITON_KERNEL_DUMP | 0 or unset | Enables or disables the dumping of Triton kernels. When enabled, Triton saves the generated kernel code (IR at each compilation stage and final PTX) to a specified directory. | 0: Do not enable<br>1: Enable | |
| **Debugging & Logging** | TRITON_DUMP_DIR | ~/.triton/dump | Specifies the directory for saving Triton kernel dump files. The directory where IR and PTX are saved when `TRITON_KERNEL_DUMP=1`. | "path": Save path | |
| **Debugging & Logging** | TRITON_DEVICE_PRINT | 0 or unset | When set to `1` or `true` (`TRUE` will be converted to `true`), enables the `tl.device_print` feature. Important note: This feature uses a GM buffer (whose pointer is passed to the kernel). | 0: Do not enable<br>1: Enable `tl.device_print` feature | The GM buffer per thread is a maximum of 16KB; content exceeding this limit will be discarded. This value is currently fixed and will be adjustable via an environment variable in the future. |
| **Debugging & Logging** | TRITON_MEMORY_DISPLAY | 0 or unset | Controls whether a JSON file for memory usage is generated. When `TRITON_MEMORY_DISPLAY=1`, saves the `memory_info_aic/aiv.json` file to the current directory. | 0: Do not enable<br>1: Enable | |
| **Compilation Control** | TRITON_ALWAYS_COMPILE | 0 or unset | Controls whether Triton forces recompilation of kernels on every run instead of using cached versions. By default, Triton caches previously compiled kernels (based on parameters and configuration) to improve performance. When set to 1, Triton ignores the cache and recompiles kernels every time, which is useful for debugging or testing new compiler features. | 0: Do not enable<br>1: Recompile all kernels on every run | |
| **Compilation Control** | DISABLE_LLVM_OPT | 0 or unset | When set to 1, disables optimization steps during LLVM compilation (LLVM optimizations for `make_llir` and `make_ptx`). When set to a string, it is parsed as a list of LLVM optimization flags to disable. For example, using `DISABLE_LLVM_OPT="disable-lsr"` disables loop strength reduction (which can cause up to 10% performance variation in some kernels with register pressure). | 0: LLVM optimizations are enabled<br>1: Disable optimization steps during LLVM compilation (LLVM optimizations for `make_llir` and `make_ptx`)<br><list>:"disable-lsr": Disable loop strength reduction</list>| |
| **Compilation Control** | MLIR_ENABLE_TIMING | 0 or unset | Enables or disables timing statistics during MLIR compilation. | 0: Do not enable<br>1: Enable | |
| **Compilation Control** | LLVM_ENABLE_TIMING | 0 or unset | Enables or disables timing statistics during LLVM compilation. | 0: Do not enable<br>1: Enable | |
| **Compilation Control** | TRITON_DEFAULT_FP_FUSION | 1 (enabled) | Controls whether floating-point operation fusion optimization is enabled by default, overriding the default fusion behavior (e.g., mul+add -> fma). | 0: Do not enable<br>1: Enable | |
| **Compilation Control** | TRITON_KERNEL_OVERRIDE | 0 or unset | Enables or disables the Triton kernel override feature, allowing user-specified external files (IR/PTX, etc.) to override the default generated kernel code at the beginning of each compilation stage. | 0: Do not enable<br>1: Enable | |
| **Compilation Control** | TRITON_OVERRIDE_DIR | ~/.triton/override | Specifies the directory for finding Triton kernel override files. The directory from which IR/PTX files are loaded when `TRITON_KERNEL_OVERRIDE=1`. | "path": Save path | |
| **Compilation Control** | TRITON_ASCEND_COMPILE_SPEED_OPT | 0 or unset | Controls whether the JIT compiler skips subsequent compilation stages after a kernel compilation failure. Set to `1` to skip (default `0` to continue trying). | 0: Continue trying<br>1: Skip | |
| **Compilation Control** | TRITON_COMPILE_ONLY | 0 or unset | Used with `remote_launch`, compiles only without running. | 0: Do not enable<br>1: Enable | |
| **Compilation Control** | TRITON_DISABLE_FFTS | 0 or unset | Whether to disable FFTS. | 0: Enable<br>1: Disable | |
| **Compilation Control** | TRITON_DISABLE_PRECOMPILE | 0 or unset | Whether to disable precompilation.                                                                                                                                                                                                                                                                                  | 0: Enable precompilation<br>1: Disable precompilation                                                                               | |
| **Execution & Scheduling** | TRITON_ALL_BLOCKS_PARALLEL | 0 or unset | Enables or disables automatic optimization of the logical core count based on the physical core count, only applicable when logical cores can run in parallel. When the logical core count exceeds the physical core count, enabling this optimization causes the compiler to automatically adjust the logical core count to the physical core count, reducing scheduling overhead; enabling this allows grid > 65535. Limitation: The logic of the Triton kernel must be insensitive to execution order to enable this option; otherwise, it may cause deadlocks. The per-kernel option `enable_auto_blockify` (see `architecture_difference.md`) takes precedence over this environment variable when explicitly set; the environment variable only acts as a default value for kernels that do not have `enable_auto_blockify` set. | 0: Do not enable<br>1: Enable | |
| **Execution & Scheduling** | TRITON_ENABLE_TASKQUEUE | 1 | Whether to enable task_queue. | 0: Do not enable<br>1: Enable | |
| **Execution & Scheduling** | TRITON_ENABLE_SANITIZER | 0 or unset | Whether to enable the SANITIZER. | 0: Do not enable<br>1: Enable | |
| **Execution & Scheduling** | ENABLE_PRINT_UB_BITS | 0 or unset | When enabled, allows obtaining the current UB usage, for use by inductor. | 0: Do not enable<br>1: Enable | |
| **Execution & Scheduling** | NPU_DEVICE_LIMIT | unset | User sets the maximum aicore and vector_core for operator runtime. Format: numbers separated by commas. For example: 14,28. | No default value | |
| **Other** | TRITON_BENCH_METHOD | unset | When using Ascend NPU, switches `do_bench` in `testing.py` to `do_bench_npu` (requires `INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE = 1`). When set to `default`, the original `do_bench` function is called even if an NPU is available. | "npu": Switch to `do_bench_npu` | |
| **Other** | TRITON_REMOTE_RUN_CONFIG_PATH | path | Specifies the configuration path for remote execution. | Provide the path directly | |


## Compiler Options


Compilation options are used to control the compilation strategy of a single Triton kernel, and can be passed via `triton.Config`, Autotune parameters, or kernel launch meta-parameters.


### Compiler Option Usage Examples


For example, `multibuffer` can be passed directly during kernel launch:


```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024, multibuffer=True)
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn((4096,), device="npu", dtype=torch.float32)
    y = torch.randn((4096,), device="npu", dtype=torch.float32)
    out = add(x, y)
    torch.npu.synchronize()
    print(out[:4])
```


### Compiler Options Reference Table


Compiler Option Configuration Reference Table:


| Category | Compilation Option | Default/Allowed Values | Description | Configuration Notes |
|----------|-------------------|------------------------|-------------|-------------------|
| **General Pipeline** | `multibuffer` | `True` (default), `False` | Enable or disable ping-pong/double buffer pipeline. Enabled by default. | `triton.Config` or launch meta-parameter |
| **CV Fusion** | `enable_auto_bind_sub_block` | `None`, `True`, `False` | Enable or disable automatic sub-block binding. | `triton.Config` or launch meta-parameter |
| **CV Fusion** | `enable_hivm_auto_cv_balance` | `None`, `True`, `False` | Enable or disable automatic CV balance. | `triton.Config` or Autotune parameter |
| **CV Fusion/Synchronization** | `sync_solver` | `None`, `True`, `False` | Enable or disable the HIVM synchronization solver. | `triton.Config` or launch meta-parameter |
| **Synchronization** | `unit_flag` | `None`, `True`, `False` | Synchronization optimization related to Cube data transfer out. | `triton.Config` or Autotune parameter |
| **Synchronization** | `inject_barrier_all` | `None`, `True`, `False` | Enable or disable automatic injection of barrier synchronization. | `triton.Config` or launch meta-parameter |
| **Synchronization** | `inject_block_all` | `None`, `True`, `False` | Enable or disable automatic injection of block synchronization. | `triton.Config` or launch meta-parameter |
| **Multi-Buffer Scope** | `limit_auto_multi_buffer_only_for_local_buffer` | `None`, `True`, `False` | Restrict automatic multi-buffer to only apply to local buffers. | `triton.Config` or Autotune parameter |
| **Multi-Buffer Scope** | `limit_auto_multi_buffer_of_local_buffer` | `None`, `"no-limit"`, `"no-l0c"` | Configure the scope of automatic multi-buffer for local buffers. | `triton.Config` or Autotune parameter |
| **Workspace** | `set_workspace_multibuffer` | `None`, `2`, `4` | Configure the workspace multi-buffer level. | `triton.Config` or Autotune parameter |
| **CV Fusion Tiling** | `tile_mix_vector_loop` | `None`, `2`, `4`, `8` | Configure the number of partitions for the Vector loop. | `triton.Config` or Autotune parameter |
| **CV Fusion Tiling** | `tile_mix_cube_loop` | `None`, `2`, `4`, `8` | Configure the number of partitions for the Cube loop. | `triton.Config` or Autotune parameter |
| **CV Fusion/Synchronization** | `disable_auto_inject_block_sync` | `None`, `True`, `False` | Enable or disable automatic block sync injection. | `triton.Config` or launch meta-parameter |
| **Execution Stream** | `stream` | `None` or NPU stream identifier | Specify the NPU stream. | launch meta-parameter |
| **Compilation Pass** | `enable_linearize` | Version-dependent | Enable or disable the linearization pass. | `triton.Config` or launch meta-parameter |
| **CV Fusion/Layout** | `enable_nd2nz_on_vector` | Default `False` | Enable or disable ND to NZ layout conversion on the Vector path. | `triton.Config` or launch meta-parameter |
| **Large Grid Optimization** | `auto_blockify_size` | Default `1` | Enable or disable the AutoBlockify pass. Ignored when `TRITON_ALL_BLOCKS_PARALLEL` is not set. | launch meta-parameter or `triton.Config` |

