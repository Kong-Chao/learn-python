# 数据分析基础：NumPy 和 Pandas
# 运行前请确保安装了库：pip install numpy pandas

import numpy as np
import pandas as pd

print("========= 1. NumPy: 数值计算基础 ==========\n")

# 1. 创建数组 (Array)
# NumPy 的核心是 ndarray 对象
arr = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {arr}")

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组(矩阵):\n{matrix}")

# 2. 数组运算
print(f"\n数组加法 (每个元素+10): {arr + 10}")
print(f"数组乘法 (每个元素*2): {arr * 2}")
print(f"矩阵均值: {np.mean(matrix)}")
print(f"矩阵最大值: {np.max(matrix)}")

# 3. 快速生成数据
zeros = np.zeros((2, 3)) # 全0矩阵
random_nums = np.random.rand(3, 3) # 3x3 随机矩阵 (0-1之间)
print(f"\n随机矩阵:\n{random_nums}")


print("\n========= 2. Pandas: 数据处理神器 ==========\n")

# 1. Series (序列) - 类似于带索引的一维数组
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(f"Series数据:\n{s}")
print(f"获取 'b' 的值: {s['b']}")

# 2. DataFrame (数据框) - 类似于 Excel 表格
# 创建测试数据
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'City': ['New York', 'Paris', 'London', 'Tokyo'],
    'Salary': [50000, 60000, 45000, 80000]
}

df = pd.DataFrame(data)
print(f"\nDataFrame数据预览:\n{df}")

# 3. 数据筛选与查询
print("\n--- 数据筛选 ---")
# 筛选年龄大于 25 的人
older_than_25 = df[df['Age'] > 25]
print(f"年龄大于25岁的人:\n{older_than_25}")

# 4. 统计分析
print("\n--- 统计信息 ---")
print(f"平均年龄: {df['Age'].mean()}")
print(f"薪资描述统计:\n{df['Salary'].describe()}")

# 5. 简单的数据操作
print("\n--- 数据操作 ---")
# 添加新列
df['Bonus'] = df['Salary'] * 0.1
print(f"添加奖金列后:\n{df.head()}") # head() 默认显示前5行

# 按城市分组计算平均薪资
city_salary = df.groupby('City')['Salary'].mean()
print(f"\n各城市平均薪资:\n{city_salary}")

# 6. 文件读写 (示例代码，实际运行需要文件)
# df.to_csv('data.csv', index=False)  # 保存为 CSV
# df = pd.read_csv('data.csv')        # 读取 CSV
# df.to_excel('data.xlsx')            # 保存为 Excel (需要 openpyxl 库)
