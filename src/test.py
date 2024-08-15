import torch
import numpy as np
from core.tensor import TritonTensor

torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
tensor_from_torch_0 = TritonTensor(torch_tensor)

print(tensor_from_torch_0)

torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
tensor_from_torch_1 = TritonTensor(torch_tensor)

print(tensor_from_torch_1)

tensor_from_torch_2 = tensor_from_torch_0 + tensor_from_torch_1
print(tensor_from_torch_2)
