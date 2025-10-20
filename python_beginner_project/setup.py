from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="myapp",
    version="0.1.0",
    author="Python初学者",
    author_email="example@example.com",
    description="一个Python新手示例项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/python-beginner-project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # 在这里添加项目依赖
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.11",
        ],
    },
)