# Polars 全攻略：高性能数据处理的新选择
# 注意：运行此脚本需要安装 polars 和 pyarrow 库
# 安装命令：pip install polars pyarrow

import polars as pl
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def print_df(title, df):
    print(f"\n--- {title} ---")
    print(df)

print("========= Polars 全攻略学习指南 ==========\n")
print(f"Polars 版本: {pl.__version__}\n")

# ==========================================
# 第一部分：创建 DataFrame
# ==========================================
print_section("1. 创建 DataFrame")

# 1. 从字典创建
data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 35, 40, 25],
    "salary": [50000, 60000, 70000, 80000, 55000],
    "join_date": [
        datetime(2020, 1, 1),
        datetime(2019, 5, 15),
        datetime(2021, 3, 10),
        datetime(2018, 11, 23),
        datetime(2022, 1, 1)
    ]
}
df = pl.DataFrame(data)
print_df("从字典创建", df)

# 2. 从 Pandas DataFrame 转换
pdf = pd.DataFrame(data)
df_from_pd = pl.from_pandas(pdf)
print_df("从 Pandas 转换 (0开销)", df_from_pd)

# 3. 查看数据概览
print("\n--- 数据概览 (glimpse) ---")
print(df.glimpse(return_as_string=True)) # 类似于 str() 或 info()

print("\n--- 查看 Schema ---")
print(df.schema)

# ==========================================
# 第二部分：核心概念 - 表达式 (Expressions)
# ==========================================
print_section("2. 核心概念 - 表达式 (Expressions)")
print("Polars 的核心是表达式，它们是懒惰评估的，可以并行执行且内存高效。")

# 1. select: 选择列
# pl.col("name") 是一个表达式
result = df.select([
    pl.col("name"),
    pl.col("age"),
    (pl.col("salary") * 1.1).alias("salary_increased") # 计算并重命名
])
print_df("select & 计算", result)

# 2. filter: 过滤行
result = df.filter(
    (pl.col("age") > 25) & (pl.col("salary") < 80000)
)
print_df("filter (age > 25 & salary < 80000)", result)

# 3. with_columns: 添加或修改列
# 类似于 pandas 的 assign，但更高效
result = df.with_columns([
    (pl.col("age") + 1).alias("age_next_year"),
    pl.lit("IT").alias("department") # 添加常数列
])
print_df("with_columns 添加列", result)

# ==========================================
# 第三部分：分组与聚合
# ==========================================
print_section("3. 分组与聚合 (Group By & Aggregation)")

# 添加一个部门列以便演示
df_dept = df.with_columns([
    pl.Series("dept", ["HR", "IT", "HR", "IT", "Marketing"])
])

# 1. 基础分组聚合
# maintain_order=True 保持结果顺序（会稍微影响性能，但在学习时很有用）
result = df_dept.group_by("dept", maintain_order=True).agg([
    pl.col("salary").mean().alias("avg_salary"),
    pl.col("salary").max().alias("max_salary"),
    pl.col("name").count().alias("employee_count"),
    pl.col("name").first().alias("first_employee")
])
print_df("分组聚合结果", result)

# 2. 复杂聚合：在 agg 中使用表达式
result = df_dept.group_by("dept").agg([
    pl.col("salary").filter(pl.col("age") > 30).mean().alias("avg_salary_senior"), # 只计算 age > 30 的平均薪资
    pl.col("name") # 返回列表
])
print_df("复杂聚合 (含条件过滤)", result)

# ==========================================
# 第四部分：Lazy API (懒惰执行)
# ==========================================
print_section("4. Lazy API (懒惰执行)")
print("Lazy API 允许 Polars 优化查询计划，合并操作，减少内存使用。")

# 1. 转换为 LazyFrame
lf = df.lazy()

# 2. 构建查询计划 (不会立即执行)
query = (
    lf
    .filter(pl.col("age") >= 25)
    .with_columns((pl.col("salary") / 1000).alias("salary_k"))
    .group_by("age")
    .agg(pl.col("salary_k").sum())
    .sort("age")
)

print("\n--- 查询计划 (Explain) ---")
# explain() 可以看到 Polars 如何优化查询（例如谓词下推）
print(query.explain())

# 3. 执行查询 (collect)
result = query.collect()
print_df("Lazy Query 执行结果", result)

# ==========================================
# 第五部分：IO 操作 (读写文件)
# ==========================================
print_section("5. I/O 操作")

# 创建临时 CSV 文件
csv_file = "temp_polars_data.csv"
df.write_csv(csv_file)
print(f"已写入: {csv_file}")

# 1. Eager 读取 (直接读取到内存)
df_read = pl.read_csv(csv_file)
print_df("read_csv 读取", df_read.head(2))

# 2. Lazy 读取 (扫描文件，用于处理大数据)
# scan_csv 不会读取整个文件，而是返回一个 LazyFrame
lf_scan = pl.scan_csv(csv_file)
result = lf_scan.filter(pl.col("age") > 30).collect()
print_df("scan_csv + filter + collect", result)

# 清理文件
if os.path.exists(csv_file):
    os.remove(csv_file)
    print(f"已清理: {csv_file}")

# ==========================================
# 第六部分：窗口函数与时间序列
# ==========================================
print_section("6. 窗口函数与时间序列")

# 1. 窗口函数 (Over)
# 计算每个部门的平均薪资，并将其作为新列附加到原始数据
result = df_dept.with_columns([
    pl.col("salary").mean().over("dept").alias("dept_avg_salary")
])
print_df("窗口函数 (over)", result)

# 2. 时间序列重采样 (Upsample/Downsample)
# 创建一个时间序列数据
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
values = np.random.randn(10)
ts_df = pl.DataFrame({"date": dates, "value": values})

# 按 2 天进行重采样并计算均值
# dynamic grouping using group_by_dynamic is powerful for time series
result = (
    ts_df
    .sort("date")
    .group_by_dynamic("date", every="2d")
    .agg(pl.col("value").mean())
)
print_df("时间序列重采样 (2d mean)", result)

# ==========================================
# 总结
# ==========================================
print_section("总结")
print("""
Polars 的核心优势：
1. 速度快：Rust 编写，并行化处理。
2. 内存高效：Zero-copy 机制。
3. 表达式系统：优雅且功能强大。
4. Lazy API：自带查询优化器，适合处理大型数据集。

何时使用 Polars？
- 数据量较大，Pandas 内存不足或速度慢时。
- 需要高性能数据清洗和预处理时。
""")
