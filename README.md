
# TritonCL
- [English](README.md)
- [简体中文](README_CN.md)

TritonCL is a high-performance library implementing all BLAS (Basic Linear Algebra Subprograms) functions and some AI operations using the Triton deep learning compiler. This library is designed for scientific computing and AI workloads, leveraging the power of modern GPUs for efficient computations.

## Features
- **BLAS Level 1**: Vector operations (AXPY, DOT, NRM2, etc.)
- **BLAS Level 2**: Matrix-vector operations (GEMV, TRSV, etc.)
- **BLAS Level 3**: Matrix-matrix operations (GEMM, SYRK, etc.)
- **AI Operations**: Common operations in AI such as 2D Convolution, Self-Attention

## Installation

### Prerequisites
Make sure you have the following dependencies installed:
- Python 3.8+
- Triton 2.0+ (`pip install triton`)
- CUDA-enabled GPU with compatible drivers

### Install via setup.py

To install Triton-BLAS, clone the repository and run:

```bash
git clone https://github.com/your-repo/triton-blas.git
cd triton-blas
pip install .
```

Alternatively, you can install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

Once installed, you can import and use the library as follows:

```python
import triton_blas.blas_ops.level1.axpy as axpy
import triton_blas.blas_ops.level3.gemm as gemm

# Example: AXPY operation
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
alpha = 0.5
result = axpy.axpy(alpha, x, y)
print(result)

# Example: GEMM operation (Matrix Multiplication)
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = gemm.gemm(A, B)
print(C)
```

### AI Operations

```python
import triton_blas.ai_ops.conv2d as conv2d

# Example: 2D Convolution
input_tensor = ...
filter_tensor = ...
output_tensor = conv2d.conv2d(input_tensor, filter_tensor)
```

## Testing

Unit tests are provided for every implemented function. To run the tests, use the following command:

```bash
pytest tests/
```

## Benchmarks

Performance benchmarks are available in the `benchmarks/` directory. For example, to benchmark the GEMM implementation:

```bash
python benchmarks/benchmark_gemm.py
```

## Implemented Operators
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

## Contributing

Contributions are welcome! If you wish to contribute, please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a pull request

Please make sure your code is well-documented and includes relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

