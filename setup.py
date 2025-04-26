from setuptools import setup, find_packages

setup(
    name='MSLF',  # 项目名称
    version='0.1',  # 版本号
    author='Your Name',  # 作者
    description='A style transfer framework based on segmentation perception and multi-scale fusion',  # 简短描述
    long_description=open('README.md').read(),  # 项目的详细描述（通常从 README 文件中读取）
    long_description_content_type='text/markdown',  # 说明 README 的格式类型
    url='https://github.com/Wjbbbbb/MSLF',  # 项目的 GitHub 仓库链接
    packages=find_packages(),  # 自动寻找并包含所有的包
    install_requires=[  # 项目依赖的第三方库
        'torch==1.13.0+cu116',  # 根据你的要求设置相应的依赖
        'torchvision==0.14.0+cu116',
        'numpy',
        'opencv-python',
        'matplotlib',
        'scikit-image',
        'Pillow',
        'transformers',  # 如果需要其他库可以添加
    ],
    classifiers=[  # 用于分类项目类型
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # 适用的 Python 版本
)
