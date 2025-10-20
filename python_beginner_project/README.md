# Python新手项目

这是一个为Python初学者设计的示例项目，展示了典型的Python项目结构和最佳实践。

## 项目结构

```
python_beginner_project/
├── src/                    # 源代码目录
│   ├── myapp/              # 主应用包
│   │   ├── __init__.py     # 包初始化文件
│   │   ├── utils/          # 工具函数模块
│   │   ├── models/         # 数据模型模块
│   │   ├── services/       # 业务逻辑模块
│   │   └── api/            # API接口模块
│   ├── tests/              # 测试代码目录
│   └── config/             # 配置文件目录
├── docs/                   # 文档目录
├── examples/               # 示例代码目录
├── README.md               # 项目说明文件
├── setup.py                # 安装配置文件
└── requirements.txt        # 依赖项列表
```

## 功能说明

这个项目展示了Python项目的基本结构和组织方式，适合初学者学习如何组织自己的Python代码。

## 安装

```bash
pip install -e .
```

## 使用示例

```python
from myapp.utils.string_utils import greet

print(greet("Python初学者"))
```

## 运行测试

```bash
pytest
```