import torch
import numpy as np
import triton
import triton.language as tl

import sys
print(sys.path)

class TritonTensor:
    def __init__(self, data, device='cuda'):
        """
        TritonTensor 初始化方法.

        参数:
        - data: 一个包含数值的 NumPy 数组或 PyTorch 张量.
        - device: 设备标识，默认是 'cuda' (GPU).
        """
        if isinstance(data, np.ndarray):
            # 从 NumPy 数组创建 PyTorch 张量
            self.data = torch.tensor(data, device=device)
        elif isinstance(data, torch.Tensor):
            # 从 PyTorch 张量创建 TritonTensor
            self.data = data.to(device)
        else:
            raise TypeError("Unsupported data type for TritonTensor.")

        self.shape = self.data.shape
        self.device = device

    def to_numpy(self):
        """
        将 TritonTensor 转换为 NumPy 数组.

        返回:
        - ndarray: 转换后的 NumPy 数组.
        """
        return self.data.cpu().numpy()

    @staticmethod
    def from_numpy(ndarray, device='cuda'):
        """
        从 NumPy 数组创建 TritonTensor.

        参数:
        - ndarray: 要转换的 NumPy 数组.
        - device: 设备标识，默认是 'cuda' (GPU).

        返回:
        - TritonTensor: 新创建的 TritonTensor 实例.
        """
        return TritonTensor(torch.tensor(ndarray, device=device))

    def __repr__(self):
        return f"TritonTensor(shape={self.shape}, device={self.device})"
    
    def __str__(self):
        return f"TritonTensor(shape={self.shape}, device={self.device}, data={self.data.cpu().numpy()})"
    
    def __getitem__(self, index):
        """
        实现张量下标功能.

        参数:
        - index: 用于索引的下标，支持单个或多个索引.

        返回:
        - TritonTensor: 被索引后的新 TritonTensor 实例.
        """
        if isinstance(index, tuple):
            result_data = self.data[index]
        else:
            result_data = self.data[index]
        return TritonTensor(result_data, device=self.device)

    def __add__(self, other):
        """
        实现张量加法运算.

        参数:
        - other: 另一个 TritonTensor 或标量.

        返回:
        - TritonTensor: 相加后的新张量.
        """
        if isinstance(other, TritonTensor):
            result_data = self.data + other.data
        else:
            result_data = self.data + other
        return TritonTensor(result_data, device=self.device)

    def __sub__(self, other):
        """
        实现张量减法运算.

        参数:
        - other: 另一个 TritonTensor 或标量.

        返回:
        - TritonTensor: 相减后的新张量.
        """
        if isinstance(other, TritonTensor):
            result_data = self.data - other.data
        else:
            result_data = self.data - other
        return TritonTensor(result_data, device=self.device)

    def __mul__(self, other):
        """
        实现张量乘法运算.

        参数:
        - other: 另一个 TritonTensor 或标量.

        返回:
        - TritonTensor: 相乘后的新张量.
        """
        if isinstance(other, TritonTensor):
            result_data = self.data * other.data
        else:
            result_data = self.data * other
        return TritonTensor(result_data, device=self.device)

    def __truediv__(self, other):
        """
        实现张量除法运算.

        参数:
        - other: 另一个 TritonTensor 或标量.

        返回:
        - TritonTensor: 相除后的新张量.
        """
        if isinstance(other, TritonTensor):
            result_data = self.data / other.data
        else:
            result_data = self.data / other
        return TritonTensor(result_data, device=self.device)

    def transpose(self):
        """
        转置张量.

        返回:
        - TritonTensor: 转置后的新张量.
        """
        transposed_data = self.data.T
        return TritonTensor(transposed_data, device=self.device)

    def norm(self, ord=None):
        """
        计算张量的范数.

        参数:
        - ord: 范数类型，默认 None 表示 Frobenius 范数.

        返回:
        - float: 张量的范数值.
        """
        return torch.norm(self.data, p=ord).item()
