# NumPy 全攻略：从入门到精通 (完整版)
# 注意：运行此脚本需要安装 numpy 库
# 安装命令：pip install numpy

import sys
import os

try:
    import numpy as np
except ImportError:
    print("错误：未找到 numpy 库。请运行 pip install numpy 安装")
    sys.exit(1)

print("========= NumPy 全攻略学习指南 (完整版) ==========\n")
print(f"NumPy 版本: {np.__version__}\n")

# ==========================================
# 第一部分：数组创建与基础属性
# ==========================================
print("=== 1. 数组创建与基础属性 ===")

# 1.1 基础创建
print("--- 基础创建 ---")
arr_list = np.array([1, 2, 3, 4, 5], dtype=np.float32) # 指定类型
print(f"列表转数组(指定float32): {arr_list}, 类型: {arr_list.dtype}")

# 1.2 常用生成函数
print("\n--- 常用生成函数 ---")
print(f"全0数组 (zeros): {np.zeros(3)}")
print(f"全1数组 (ones): {np.ones((2, 2))}")
print(f"空数组 (empty, 内存残留值): {np.empty((2, 2))}")
print(f"单位矩阵 (eye): \n{np.eye(3)}")
print(f"范围 (arange): {np.arange(0, 10, 2)}")
print(f"等分 (linspace): {np.linspace(0, 1, 5)}") # 0到1之间5个点
print(f"对数等分 (logspace): {np.logspace(0, 2, 3)}") # 10^0 到 10^2

# 1.3 属性检查
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n--- 数组属性 ---")
print(f"形状 (shape): {arr.shape}")
print(f"维度 (ndim): {arr.ndim}")
print(f"元素数 (size): {arr.size}")
print(f"元素字节大小 (itemsize): {arr.itemsize}")
print(f"总字节大小 (nbytes): {arr.nbytes}")

# ==========================================
# 第二部分：数据类型与转换
# ==========================================
print("\n=== 2. 数据类型 (Data Types) ===")
# NumPy 支持比 Python 更多的数字类型
arr_int = np.array([1, 2, 3])
arr_float = arr_int.astype(np.float64) # 类型转换核心方法
print(f"整型转浮点型: {arr_float}, 类型: {arr_float.dtype}")

arr_str = np.array(['1.1', '2.2', '3.3'])
arr_num = arr_str.astype(float)
print(f"字符串转数字: {arr_num}")

# ==========================================
# 第三部分：索引与切片 (核心)
# ==========================================
print("\n=== 3. 索引与切片 (Indexing & Slicing) ===")
a = np.arange(10)

# 3.1 基础切片
print(f"切片 [2:5]: {a[2:5]}")
print(f"切片并修改: ", end="")
a_slice = a[0:3]
a_slice[0] = 999 # 注意：切片是视图，修改会影响原数组！
print(f"原数组也被修改了: {a}")
a[0] = 0 # 还原

# 3.2 布尔索引 (Boolean Indexing) - 筛选神器
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will'])
data = np.random.randn(5, 4)
print(f"\n--- 布尔索引 ---")
print(f"Names: {names}")
mask = (names == 'Bob') | (names == 'Will')
print(f"筛选 Bob 或 Will 对应的数据行:\n{data[mask]}")
data[data < 0] = 0 # 将所有负数置为0
print(f"将负数置0后的数据:\n{data}")

# 3.3 花式索引 (Fancy Indexing) - 使用整数数组索引
print(f"\n--- 花式索引 ---")
arr = np.empty((8, 4))
for i in range(8): arr[i] = i
print(f"原始矩阵:\n{arr}")
# 选取第4, 3, 0, 6行
print(f"选取特定行 [4, 3, 0, 6]:\n{arr[[4, 3, 0, 6]]}")
# 选取 (1,0), (5,3), (7,1) 处的元素
print(f"选取特定坐标元素:\n{arr[[1, 5, 7], [0, 3, 1]]}")

# ==========================================
# 第四部分：数组运算与通用函数 (ufunc)
# ==========================================
print("\n=== 4. 运算与通用函数 (ufunc) ===")
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# 4.1 基础运算
print(f"加法: {x + y}")
print(f"乘法: {x * y}")
print(f"幂运算: {x ** 2}")

# 4.2 常用 ufunc
print(f"平方根 (sqrt): {np.sqrt(x)}")
print(f"指数 (exp): {np.exp(x)}")
print(f"对数 (log): {np.log(x)}")
# 二元 ufunc
print(f"最大值比较 (maximum): {np.maximum(x, np.array([2, 2, 2]))}") # 逐元素比较

# 4.3 广播机制 (Broadcasting) 进阶
print(f"\n--- 广播机制 ---")
arr_3x3 = np.ones((3, 3))
row = np.arange(3)
col = np.arange(3).reshape(3, 1)
print(f"3x3 + 1x3 (行广播):\n{arr_3x3 + row}")
print(f"3x3 + 3x1 (列广播):\n{arr_3x3 + col}")

# ==========================================
# 第五部分：统计与聚合
# ==========================================
print("\n=== 5. 统计与聚合 ===")
arr = np.random.randn(5, 4)
print(f"均值 (mean): {arr.mean()}")
print(f"标准差 (std): {arr.std()}")
print(f"求和 (sum): {arr.sum()}")
print(f"每列求和 (axis=0): {arr.sum(axis=0)}")
print(f"每行求和 (axis=1): {arr.sum(axis=1)}")
print(f"累加 (cumsum): {np.array([1,2,3]).cumsum()}")

# 唯一值与集合逻辑
names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
print(f"唯一值 (unique): {np.unique(names)}")

# ==========================================
# 第六部分：数组变形与合并
# ==========================================
print("\n=== 6. 数组变形与合并 ===")
arr = np.arange(12).reshape(3, 4)

# 6.1 变形
print(f"展平 (ravel, 返回视图): {arr.ravel()}")
print(f"展平 (flatten, 返回副本): {arr.flatten()}")
print(f"转置 (T): \n{arr.T}")

# 6.2 合并与分割
arr1 = np.array([[1, 2, 3]])
arr2 = np.array([[4, 5, 6]])
print(f"垂直合并 (vstack): \n{np.vstack((arr1, arr2))}")
print(f"水平合并 (hstack): \n{np.hstack((arr1.T, arr2.T))}")
# concatenate 是更通用的合并方法
print(f"指定轴合并 (concatenate axis=0): \n{np.concatenate([arr1, arr2], axis=0)}")

# ==========================================
# 第七部分：排序与搜索
# ==========================================
print("\n=== 7. 排序与搜索 ===")
r_arr = np.random.randn(5)
print(f"原数组: {r_arr}")
r_arr.sort() # 就地排序
print(f"排序后: {r_arr}")

# argsort 返回排序后的索引
x = np.array([3, 1, 4, 2])
idx = np.argsort(x)
print(f"argsort索引: {idx}, 使用索引排序: {x[idx]}")

# where (条件选择)
cond = np.array([True, False, True, False])
res = np.where(cond, 1, -1) # True选1, False选-1
print(f"where 条件选择: {res}")
# where 查找索引
print(f"大于2的元素索引: {np.where(x > 2)}")

# ==========================================
# 第八部分：线性代数
# ==========================================
print("\n=== 8. 线性代数 (Linear Algebra) ===")
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print(f"矩阵乘法 (dot): \n{x.dot(y)}") # 或 np.dot(x, y) 或 x @ y

from numpy.linalg import inv, qr, eig
mat = np.random.randn(3, 3)
# 只有方阵且非奇异才能求逆，这里仅演示
# print(f"矩阵求逆: \n{inv(mat)}") 
print(f"对角线元素 (diag): {np.diag(mat)}")

# ==========================================
# 第九部分：缺失值处理 (NaN)
# ==========================================
print("\n=== 9. 缺失值处理 (NaN) ===")
arr_nan = np.array([1, 2, np.nan, 4])
print(f"含有NaN的数组: {arr_nan}")
print(f"判断NaN (isnan): {np.isnan(arr_nan)}")
# 普通 sum 会得到 nan
print(f"普通求和: {arr_nan.sum()}")
# 安全求和
print(f"忽略NaN求和 (nansum): {np.nansum(arr_nan)}")
print(f"忽略NaN均值 (nanmean): {np.nanmean(arr_nan)}")

# ==========================================
# 第十部分：随机数进阶
# ==========================================
print("\n=== 10. 随机数进阶 ===")
np.random.seed(42) # 设置随机种子，保证结果可复现
print(f"正态分布 (normal): {np.random.normal(size=(2,2))}")
print(f"均匀分布 (uniform): {np.random.uniform(0, 10, 3)}")
print(f"随机排列 (permutation): {np.random.permutation(5)}")
# 随机采样
choices = np.random.choice([10, 20, 30, 40], size=5, p=[0.1, 0.4, 0.4, 0.1])
print(f"按概率采样: {choices}")

# ==========================================
# 第十一部分：文件输入输出
# ==========================================
print("\n=== 11. 文件输入输出 ===")
arr_save = np.arange(10)
# 保存二进制文件 .npy
np.save('some_array', arr_save)
print("数组已保存为 some_array.npy")
# 读取
arr_load = np.load('some_array.npy')
print(f"读取数组: {arr_load}")

# 保存多个数组 .npz
np.savez('array_archive.npz', a=arr_save, b=arr_save)

# 保存文本文件 (如csv)
np.savetxt('array_ex.txt', arr_save.reshape(2, 5), delimiter=',')
print("数组已保存为 array_ex.txt (文本格式)")

# 清理文件
for f in ['some_array.npy', 'array_archive.npz', 'array_ex.txt']:
    if os.path.exists(f):
        os.remove(f)

print("\n========= 总结 ==========")
print("这份指南涵盖了 NumPy 95% 的常用功能。")
print("掌握这些，你已经可以从容应对绝大多数数据分析和机器学习中的数据预处理任务了！")
