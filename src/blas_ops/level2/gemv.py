import torch
import triton
import triton.language as tl

import sys
from utils import tools

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_autotune_config():
    return get_cuda_autotune_config()

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'K'],
)

@triton.jit
def gemv_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_am, stride_ak,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """
    Kernel for computing the gemv C = A x b
    A has shape (M, K), b has shape (K, 1) and C has shape (M, 1)
    """
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    num_pid_in_group = GROUP_SIZE_M
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = tl.arange(0, 16) % 1
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + offs_k[:, None] * stride_ak + offs_bn[None, :]
    
    accumulator = tl.zeros((BLOCK_SIZE_M, 16), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_ak
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tl.arange(0, 16)
    c_ptrs = c_ptr + offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < 1)
    tl.store(c_ptrs, c, mask=c_mask)

def gemv(a, b, activation=""): 
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    c = torch.empty((M, 1), device=a.device, dtype=torch.float16)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)
    gemv_kernel[grid](
        a, b, c,
        M, K,
        a.stride(0), a.stride(1),
        ACTIVATION=activation
    )
    return c

# torch.manual_seed(0)
# a = torch.randn((256, 256), device='cuda', dtype=torch.float16)
# b = torch.randn((256, 1), device='cuda', dtype=torch.float16)
# triton_output = gemv(a, b)
# torch_output = torch.matmul(a, b)

# print(f"triton_output_with_fp16_inputs={triton_output}")
# print(f"torch_output_with_fp16_inputs={torch_output}")

# rtol = 0
# if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")

# ref_lib = 'cuBLAS'

# configs = []
# configs.append(
#     triton.testing.Benchmark(
#         x_names=["M", "K"],
#         x_vals=[[128 * i, 128 * i] for i in range(2, 32)],
#         line_arg="provider",
#         line_vals=[ref_lib.lower(), "triton"],  # Label name for the lines
#         line_names=[ref_lib, "Triton"],  # Line styles
#         styles=[("green", "-"), ("blue", "-")],
#         ylabel="TFLOPS",  # Label name for the y-axis
#         plot_name="matmul-performance-" + "fp16",
#         args={}
#     )
# )

# @triton.testing.perf_report(configs)
# def benchmark(M, K, provider):
#     a = torch.randn((M, K), device='cuda', dtype=torch.float16)
#     b = torch.randn((K, 1), device='cuda', dtype=torch.float16)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == ref_lib.lower():
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemv(a, b), quantiles=quantiles)
#     perf = lambda ms: 2 * M * K * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)

# benchmark.run(show_plots=True, print_data=True)