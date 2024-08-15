import torch
import triton
import triton.language as tl

import sys
from src.utils import tools

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
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
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + offs_k[:, None] * stride_ak
    
    accumulator = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_ak
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + offs_cm[:, None]
    c_mask = (offs_cm[:, None] < M)
    tl.store(c_ptrs, c, mask=c_mask)

def gemv(a, b, activation=""): 
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    c = torch.empty((M, 1), device=a.device, dtype=torch.float16)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']))
    gemv_kernel[grid](
        a, b, c,
        M, K,
        a.stride(0), a.stride(1),
        ACTIVATION=activation
    )
    return c

torch.manual_seed(0)
a = torch.randn((256, 256), device='cuda', dtype=torch.float16)
b = torch.randn((256, 1), device='cuda', dtype=torch.float16)
triton_output = gemv(a, b)
torch_output = torch.matmul(a, b)

print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")

rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
