
# TritonCL
- [English](README.md)
- [简体中文](README_CN.md)

TritonCL 是一个高性能库，使用 Triton 深度学习编译器实现了全部 BLAS（基础线性代数子程序）函数和一些 AI 操作。该库为科学计算和 AI 工作负载而设计，利用现代 GPU 的计算能力来实现高效计算。

## 功能
- **BLAS Level 1**: 向量操作（AXPY，DOT，NRM2 等）
- **BLAS Level 2**: 矩阵-向量操作（GEMV，TRSV 等）
- **BLAS Level 3**: 矩阵-矩阵操作（GEMM，SYRK 等）
- **AI 操作**: 2D 卷积，自注意力机制等常见的 AI 操作

## 安装

### 前提条件
请确保已安装以下依赖项：
- Python 3.8+
- Triton 2.0+ (`pip install triton`)
- 具有兼容驱动程序的 CUDA 支持 GPU

### 通过 setup.py 安装

要安装 Triton-BLAS，请克隆仓库并运行：

```bash
git clone https://github.com/your-repo/triton-blas.git
cd triton-blas
pip install .
```

或者，你也可以通过 `requirements.txt` 安装依赖项：

```bash
pip install -r requirements.txt
```

## 使用方法

安装后，你可以按以下方式导入并使用库：

```python
import triton_blas.blas_ops.level1.axpy as axpy
import triton_blas.blas_ops.level3.gemm as gemm

# 示例：AXPY 操作
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
alpha = 0.5
result = axpy.axpy(alpha, x, y)
print(result)

# 示例：GEMM 操作（矩阵乘法）
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = gemm.gemm(A, B)
print(C)
```

### AI 操作

```python
import triton_blas.ai_ops.conv2d as conv2d

# 示例：2D 卷积操作
input_tensor = ...
filter_tensor = ...
output_tensor = conv2d.conv2d(input_tensor, filter_tensor)
```

## 测试

每个实现的函数都提供了单元测试。要运行测试，请使用以下命令：

```bash
pytest tests/
```

## 性能基准

性能基准脚本位于 `benchmarks/` 目录中。 例如，要对 GEMM 实现进行基准测试：

```bash
python benchmarks/benchmark_gemm.py
```

## 已实现算子
### BLAS
| Routine            | Signature                                 |   Implemented |
|--------------------|--------------------------------------------|:--------------:|
| xROTG              | A, B, C, S                                | |
| xROTMG             | D1, D2, A, B, PARAM                        | |
| xROT               | N, X, INCX, Y, INCY, C, S                  | |
| xROTM              | N, X, INCX, Y, INCY, PARAM                 | |
| xSWAP              | N, X, INCX, Y, INCY                        | |
| xSCAL              | N, ALPHA, X, INCX                          | |
| xCOPY              | N, X, INCX, Y, INCY                        | |
| xAXPY              | N, ALPHA, X, INCX, Y, INCY                | |
| xDOT               | N, X, INCX, Y, INCY                        | |
| xDOTU              | N, X, INCX, Y, INCY                        | |
| xDOTC              | N, X, INCX, Y, INCY                        | |
| xxDOT              | N, X, INCX, Y, INCY                        | |
| xNRM2              | N, X, INCX                                 | |
| xASUM              | N, X, INCX                                 | |
| xGEMV              | TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY |  <span style="color:green">&#10004;</span> |
| xGBMV              | TRANS, M, N, KL, KU, ALPHA, A, LDA, X, INCX, BETA, Y, INCY ||
| xHEMV              | UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY ||
| xHBMV              | UPLO, N, K, ALPHA, A, LDA, X, INCX, BETA, Y, INCY ||
| xHPMV              | UPLO, N, ALPHA, AP, X, INCX, BETA, Y, INCY ||
| xSYMV              | UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY ||
| xSBMV              | UPLO, N, K, ALPHA, A, LDA, X, INCX, BETA, Y, INCY ||
| xSPMV              | UPLO, N, ALPHA, AP, X, INCX, BETA, Y, INCY ||
| xTRMV              | UPLO, TRANS, DIAG, N, A, LDA, X, INCX ||
| xTBMV              | UPLO, TRANS, DIAG, N, K, A, LDA, X, INCX ||
| xTPMV              | UPLO, TRANS, DIAG, N, AP, X, INCX ||
| xTRSV              | UPLO, TRANS, DIAG, N, A, LDA, X, INCX ||
| xTBSV              | UPLO, TRANS, DIAG, N, K, A, LDA, X, INCX ||
| xTPSV              | UPLO, TRANS, DIAG, N, AP, X, INCX ||
| xGER               | M, N, ALPHA, X, INCX, Y, INCY, A, LDA ||
| xGERU              | M, N, ALPHA, X, INCX, Y, INCY, A, LDA ||
| xGERC              | M, N, ALPHA, X, INCX, Y, INCY, A, LDA ||
| xHER               | UPLO, N, ALPHA, X, INCX, A, LDA ||
| xHPR               | UPLO, N, ALPHA, X, INCX, AP ||
| xHER2              | UPLO, N, ALPHA, X, INCX, Y, INCY, A, LDA ||
| xHPR2              | UPLO, N, ALPHA, X, INCX, Y, INCY, AP ||
| xSYR               | UPLO, N, ALPHA, X, INCX, A, LDA ||
| xSPR               | UPLO, N, ALPHA, X, INCX, AP ||
| xSYR2              | UPLO, N, ALPHA, X, INCX, Y, INCY, A, LDA ||
| xGEMM              | TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC |  <span style="color:green">&#10004;</span> |
| xSYMM              | SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC ||
| xHEMM              | SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC ||
| xSYRK               | UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC ||
| xHERK              | UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC ||
| xSYR2K             | UPLO, TRANS, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ||
| xHER2K             | UPLO, TRANS, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ||
| xTRMM              | SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB ||
| xTRSM              | SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB ||

## 贡献指南

欢迎贡献！如果你想贡献代码，请按以下步骤进行：
1. Fork 此仓库
2. 创建一个功能分支 (`git checkout -b feature-name`)
3. 提交更改 (`git commit -am '添加新功能'`)
4. 推送到分支 (`git push origin feature-name`)
5. 创建一个 Pull Request

请确保你的代码有完整的文档说明并包含相关的测试。

## 许可证

此项目根据 MIT 许可证授权。请参阅 [LICENSE](LICENSE) 文件了解更多信息。

