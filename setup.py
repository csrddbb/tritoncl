from setuptools import setup, find_packages

setup(
    name='tritoncl',  # 包名称
    version='0.1',  # 包版本
    author='lidongsheng',  # 作者名称
    author_email='lidsh25@sysu.edu.cn',  # 作者邮箱
    description='使用 Triton 实现的 BLAS 库',  # 简短描述
    long_description=open('README.md').read(),  # 详细描述，通常从 README 文件读取
    long_description_content_type='text/markdown',  # 描述内容的类型
    url='https://github.com/csrddbb/tritoncl',  # 项目网址
    packages=find_packages(where='src'),  # 包的发现，指向 src 目录
    package_dir={'': 'src'},  # 指定包的根目录
    python_requires='>=3.6',  # 指定支持的 Python 版本
    classifiers=[  # 分类器，用于在 PyPI 中进行分类
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',  # 许可证
        'Operating System :: OS Independent',
    ],
    include_package_data=True,  # 包含其他文件，如数据文件
    zip_safe=False,  # 包是否可以安全地打包为 .egg 文件
)
