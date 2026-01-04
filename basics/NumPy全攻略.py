# NumPy 全攻略：从入门到精通
# 注意：运行此脚本需要安装 numpy 库
# 安装命令：pip install numpy

import sys

try:
    import numpy as np
except ImportError:
    print("错误：未找到 numpy 库。")
    print("请先在终端运行以下命令进行安装：")
    print("pip install numpy")
    print("或者如果使用的是清华源：")
    print("pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple")
    sys.exit(1)

print("========= NumPy 全攻略学习指南 ==========\n")
print(f"NumPy 版本: {np.__version__}\n")

# 1. NumPy 数组的基础 (ndarray)
print("=== 1. 创建数组的多种方式 ===")

# 从列表创建
py_list = [1, 2, 3, 4, 5]
arr_from_list = np.array(py_list)
print(f"1. 从列表创建: {arr_from_list}")

# 创建全0数组
zeros = np.zeros(5)
print(f"2. 全0数组 (zeros): {zeros}")

# 创建全1矩阵 (指定形状)
ones = np.ones((2, 3)) # 2行3列
print(f"3. 全1矩阵 (ones):\n{ones}")

# 创建范围数组 (类似 range)
range_arr = np.arange(0, 10, 2) # start, stop, step
print(f"4. 范围数组 (arange): {range_arr}")

# 创建等差数列 (指定数量)
linspace_arr = np.linspace(0, 10, 5) # 0到10之间生成5个数
print(f"5. 等差数列 (linspace): {linspace_arr}")

# 创建随机数组
rand_arr = np.random.rand(3, 3) # 0-1之间的均匀分布
randint_arr = np.random.randint(0, 100, (2, 4)) # 0-100之间的随机整数
print(f"6. 随机整数矩阵:\n{randint_arr}")

print("\n=== 2. 数组的重要属性 ===")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"数组:\n{arr}")
print(f"维度数 (ndim): {arr.ndim}")       # 2
print(f"形状 (shape): {arr.shape}")      # (2, 3)
print(f"元素总数 (size): {arr.size}")     # 6
print(f"数据类型 (dtype): {arr.dtype}")   # int32 或 int64

print("\n=== 3. 索引与切片 (Indexing & Slicing) ===")
# 一维数组切片
a = np.arange(10) # [0 1 2 3 4 5 6 7 8 9]
print(f"原始数组: {a}")
print(f"切片 [2:5]: {a[2:5]}")
print(f"切片 [::2] (步长2): {a[::2]}")

# 多维数组索引
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"\n二维数组:\n{b}")
print(f"获取第二行 (索引1): {b[1]}")
print(f"获取具体元素 (行1, 列2): {b[1, 2]}") # 结果是 7
print(f"切片子区域 (前两行, 前两列):\n{b[:2, :2]}")

# 布尔索引 (非常重要！)
print("\n--- 布尔索引 (筛选) ---")
data = np.array([1, 10, 2, 9, 3, 8])
mask = data > 5
print(f"数据: {data}")
print(f"大于5的掩码: {mask}")
print(f"筛选结果: {data[mask]}") # 选择所有大于5的数

print("\n=== 4. 数组运算与广播 (Broadcasting) ===")
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# 元素级运算 (Element-wise)
print(f"加法 (x+y): {x + y}")
print(f"乘法 (x*y): {x * y}")
print(f"平方 (x**2): {x ** 2}")

# 广播机制 (Broadcasting)
# 允许不同形状的数组进行计算
matrix = np.ones((3, 3))
row = np.array([1, 2, 3])
print(f"\n矩阵:\n{matrix}")
print(f"行向量: {row}")
print(f"广播加法 (矩阵每一行都加上向量):\n{matrix + row}")

print("\n=== 5. 统计与聚合函数 ===")
arr_stat = np.random.randn(3, 4) # 正态分布随机数
print(f"数据:\n{arr_stat}")
print(f"总和 (sum): {arr_stat.sum()}")
print(f"平均值 (mean): {arr_stat.mean()}")
print(f"标准差 (std): {arr_stat.std()}")
print(f"最大值 (max): {arr_stat.max()}")
print(f"最大值索引 (argmax): {arr_stat.argmax()}")

# 指定轴 (axis) 进行计算
# axis=0 代表列，axis=1 代表行
print(f"每列的和 (axis=0): {arr_stat.sum(axis=0)}")
print(f"每行的平均值 (axis=1): {arr_stat.mean(axis=1)}")

print("\n=== 6. 数组形状操作 (Reshape & Transpose) ===")
orig = np.arange(12)
print(f"原始一维: {orig}")

# 改变形状
reshaped = orig.reshape(3, 4)
print(f"Reshape为(3, 4):\n{reshaped}")

# 转置 (行列互换)
transposed = reshaped.T
print(f"转置后 (4, 3):\n{transposed}")

# 堆叠 (Stacking)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
vstack = np.vstack((arr1, arr2)) # 垂直堆叠
hstack = np.hstack((arr1, arr2)) # 水平堆叠
print(f"垂直堆叠:\n{vstack}")
print(f"水平堆叠: {hstack}")

print("\n=== 7. 线性代数 (Linear Algebra) ===")
# 矩阵乘法 (Dot Product)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
dot_product = np.dot(A, B) # 或者 A @ B
print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")
print(f"矩阵乘积 (A @ B):\n{dot_product}")

print("\n========= 学习总结 ==========")
print("1. NumPy 是 Python 数据科学的基石，几乎所有高级库 (Pandas, Scikit-learn, PyTorch) 都基于它。")
print("2. 核心优势：C语言编写的底层，计算速度极快；支持向量化运算，避免使用 Python 循环。")
print("3. 下一步：熟练掌握切片和形状变换 (reshape)，这是处理图像和机器学习数据的基础。")
