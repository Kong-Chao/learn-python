# Pandas 全攻略：数据分析必备神器
# 注意：运行此脚本需要安装 pandas 库
# 安装命令：pip install pandas
# 推荐同时安装 openpyxl 用于处理 Excel: pip install openpyxl

import sys
import os

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("错误：未找到 pandas 或 numpy 库。请运行 pip install pandas numpy 安装")
    sys.exit(1)

print("========= Pandas 全攻略学习指南 ==========\n")
print(f"Pandas 版本: {pd.__version__}\n")

# ==========================================
# 第一部分：核心数据结构
# ==========================================
print("=== 1. 核心数据结构: Series 与 DataFrame ===")

# 1.1 Series (一维带标签数组)
print("\n--- Series (序列) ---")
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(f"Series对象:\n{s}")
print(f"获取值 (s['a']): {s['a']}")
print(f"获取多个值 (s[['a', 'c']]):\n{s[['a', 'c']]}")

# 1.2 DataFrame (二维表格)
print("\n--- DataFrame (数据框) ---")
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [24, 27, 22, 32, 29],
    'City': ['New York', 'Paris', 'London', 'Tokyo', 'Berlin'],
    'Salary': [50000, 60000, 45000, 80000, 55000]
}
df = pd.DataFrame(data)
print(f"DataFrame预览:\n{df}")

# ==========================================
# 第二部分：数据查看与检查
# ==========================================
print("\n=== 2. 数据查看与检查 ===")
print(f"前3行 (head):\n{df.head(3)}")
print(f"后2行 (tail):\n{df.tail(2)}")
print(f"数据形状 (shape): {df.shape}") # (行数, 列数)
print(f"列名 (columns): {df.columns.tolist()}")
print(f"索引 (index): {df.index.tolist()}")

print("\n--- 数据信息 (Info) ---")
df.info() # 打印内存占用、列类型、非空计数

print("\n--- 统计描述 (Describe) ---")
print(df.describe()) # 快速查看数值列的统计信息 (均值、标准差、分位数等)

# ==========================================
# 第三部分：数据选择与索引 (核心)
# ==========================================
print("\n=== 3. 数据选择与索引 (Loc & iLoc) ===")

# 3.1 按列选择
print(f"选择单列 (返回Series):\n{df['Name'].head(2)}")
print(f"选择多列 (返回DataFrame):\n{df[['Name', 'City']].head(2)}")

# 3.2 loc: 基于标签(Label)的索引
print("\n--- loc (标签索引) ---")
# df.loc[行标签, 列标签]
print(f"获取第0行数据:\n{df.loc[0]}")
print(f"获取行0-2, 'Name'和'Age'列:\n{df.loc[0:2, ['Name', 'Age']]}") # 注意: loc切片包含结束索引!

# 3.3 iloc: 基于位置(Integer)的索引
print("\n--- iloc (位置索引) ---")
# df.iloc[行位置, 列位置]
print(f"获取第1行, 第2列的值: {df.iloc[1, 2]}") # Bob's City (Paris)
print(f"获取前3行, 前2列:\n{df.iloc[:3, :2]}") # iloc切片不包含结束索引 (Python习惯)

# 3.4 布尔索引 (条件筛选)
print("\n--- 布尔索引 (筛选) ---")
# 筛选 Age > 25
condition = df['Age'] > 25
print(f"Age > 25 的数据:\n{df[condition]}")
# 复杂条件: Age > 25 AND City startswith 'T'
complex_cond = (df['Age'] > 25) & (df['City'].str.startswith('T'))
print(f"Age > 25 且 城市T开头:\n{df[complex_cond]}")

# ==========================================
# 第四部分：数据清洗
# ==========================================
print("\n=== 4. 数据清洗 (Missing Data & Duplicates) ===")
# 创建脏数据
df_dirty = df.copy()
df_dirty.loc[5] = ['Alice', np.nan, 'New York', 50000] # 重复行 + 缺失值
df_dirty.loc[6] = ['Frank', 35, np.nan, np.nan] # 更多缺失值
print(f"脏数据预览:\n{df_dirty}")

# 4.1 缺失值处理
print("\n--- 缺失值处理 ---")
print(f"检查缺失值 (isnull):\n{df_dirty.isnull().sum()}") # 每列缺失多少个

# 方式A: 删除缺失值
print(f"删除含缺失值的行 (dropna):\n{df_dirty.dropna()}")

# 方式B: 填充缺失值
print(f"填充缺失值 (fillna):\n{df_dirty.fillna({'Age': df['Age'].mean(), 'City': 'Unknown', 'Salary': 0})}")

# 4.2 重复值处理
print("\n--- 重复值处理 ---")
print(f"是否存在重复行: {df_dirty.duplicated().any()}")
print(f"删除重复行 (keep='first'):\n{df_dirty.drop_duplicates(keep='first')}")

# ==========================================
# 第五部分：数据处理与转换
# ==========================================
print("\n=== 5. 数据处理与转换 ===")

# 5.1 增加/删除列
df['Bonus'] = df['Salary'] * 0.1 # 向量化运算
print(f"增加Bonus列:\n{df.head(2)}")
df = df.drop(columns=['Bonus']) # 删除列

# 5.2 排序
print(f"按Age降序排列:\n{df.sort_values(by='Age', ascending=False)}")

# 5.3 Apply 函数应用 (非常重要)
print("\n--- Apply 函数 ---")
def salary_level(salary):
    if salary > 60000: return 'High'
    elif salary > 50000: return 'Medium'
    else: return 'Low'

df['Level'] = df['Salary'].apply(salary_level)
print(f"应用自定义函数后:\n{df[['Name', 'Salary', 'Level']]}")

# 5.4 字符串处理 (.str 访问器)
print(f"城市大写:\n{df['City'].str.upper()}")

# ==========================================
# 第六部分：分组与聚合 (Groupby)
# ==========================================
print("\n=== 6. 分组与聚合 (Groupby) ===")
# 添加一列 Department
df['Dept'] = ['HR', 'IT', 'HR', 'IT', 'Finance']

# 6.1 基本分组
print(f"按部门分组计算平均薪资:\n{df.groupby('Dept')['Salary'].mean()}")

# 6.2 多种聚合
print("\n--- 多种聚合 (agg) ---")
agg_res = df.groupby('Dept').agg({
    'Salary': ['mean', 'max', 'min'],
    'Age': 'mean'
})
print(f"聚合结果:\n{agg_res}")

# ==========================================
# 第七部分：数据合并 (Merge & Concat)
# ==========================================
print("\n=== 7. 数据合并 (Merge & Concat) ===")
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'val2': [4, 5, 6]})

# 7.1 Merge (类似 SQL Join)
print(f"Inner Join (交集):\n{pd.merge(df1, df2, on='key', how='inner')}")
print(f"Left Join (左连接):\n{pd.merge(df1, df2, on='key', how='left')}")

# 7.2 Concat (物理拼接)
print(f"垂直拼接:\n{pd.concat([df1, df2], axis=0, ignore_index=True)}")

# ==========================================
# 第八部分：时间序列 (Time Series)
# ==========================================
print("\n=== 8. 时间序列 (Time Series) ===")
# 创建时间范围
dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
df['Date'] = dates
print(f"带有时间列的DF:\n{df[['Name', 'Date']]}")

# 时间属性访问器 (.dt)
print(f"获取星期几:\n{df['Date'].dt.day_name()}")
print(f"获取月份:\n{df['Date'].dt.month}")

# ==========================================
# 第九部分：文件输入输出 (I/O)
# ==========================================
print("\n=== 9. 文件输入输出 (I/O) ===")
csv_file = 'temp_pandas_learn.csv'
excel_file = 'temp_pandas_learn.xlsx'

# 9.1 写入
df.to_csv(csv_file, index=False)
print(f"已保存 CSV: {csv_file}")

# 需要 openpyxl 库支持 Excel
try:
    df.to_excel(excel_file, index=False)
    print(f"已保存 Excel: {excel_file}")
except ImportError:
    print("提示：安装 openpyxl 后可支持 Excel 导出 (pip install openpyxl)")

# 9.2 读取
df_read = pd.read_csv(csv_file)
print(f"读取 CSV 前2行:\n{df_read.head(2)}")

# 清理临时文件
import os
if os.path.exists(csv_file): os.remove(csv_file)
if os.path.exists(excel_file): os.remove(excel_file)

print("\n========= 总结 ==========")
print("Pandas 是 Python 数据分析的核心，熟练掌握 DataFrame 的索引(loc/iloc)、清洗(dropna/fillna)、")
print("分组(groupby)和合并(merge)操作，你就能处理 90% 的结构化数据任务！")
