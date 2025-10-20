import os

# 定义项目根目录
project_root = 'python_beginner_project'

# 定义目录结构
directories = [
    'src/myapp/utils',
    'src/myapp/models', 
    'src/myapp/services',
    'src/myapp/api',
    'src/tests',
    'src/config',
    'docs',
    'examples'
]

# 创建目录结构
for directory in directories:
    path = os.path.join(project_root, directory)
    os.makedirs(path, exist_ok=True)
    print(f"创建目录: {path}")

print("项目结构创建完成!")