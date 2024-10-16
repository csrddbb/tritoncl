import torch
import sys
import os

from blas_ops.level2 import gemv

import triton_blas as blas

M, K = 128, 64
A = torch.rand((M, K), device='cuda', dtype=torch.float16)
b = torch.rand((K, 1), device='cuda', dtype=torch.float16)

result = gemv.gemv(A, b)

print("MatA :", A)
print("Vec :", b)
print("Result: A * b = ", result)
