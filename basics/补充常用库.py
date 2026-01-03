# Python 补充常用库指南
# 这些库也是 Python 开发中非常常用的标准库

print("========= Python 补充常用库 ==========\n")

# 1. pathlib - 面向对象的文件系统路径
# 相比 os.path，pathlib 提供了更加直观和面向对象的操作方式
print("\n=== 1. pathlib - 面向对象的文件系统路径 ===")
from pathlib import Path

# 获取当前目录
current_path = Path.cwd()
print(f"当前路径: {current_path}")

# 路径拼接
new_path = current_path / "test_folder" / "test.txt"
print(f"拼接路径: {new_path}")

# 检查文件/目录是否存在
print(f"当前脚本是否存在: {Path(__file__).exists()}")

# 获取文件名和后缀
print(f"脚本文件名: {Path(__file__).name}")
print(f"脚本后缀: {Path(__file__).suffix}")

# 2. logging - 日志记录
# 在实际开发中，使用 logging 比 print 更专业，可以控制日志级别和输出格式
print("\n=== 2. logging - 日志记录 ===")
import logging

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.debug("这是一条调试信息 (debug)")      # 默认级别是 WARNING，所以这条不会显示
logging.info("这是一条普通信息 (info)")        # 配置为 INFO 后，这条会显示
logging.warning("这是一条警告信息 (warning)")
logging.error("这是一条错误信息 (error)")
logging.critical("这是一条严重错误信息 (critical)")

# 3. csv - CSV文件处理
# CSV 是一种通用的数据交换格式
print("\n=== 3. csv - CSV文件处理 ===")
import csv

# 写入 CSV
data = [
    ['Name', 'Age', 'City'],
    ['Alice', 28, 'New York'],
    ['Bob', 22, 'Los Angeles'],
    ['Charlie', 35, 'Chicago']
]

csv_file = "temp_users.csv"
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(data)
print(f"已写入 CSV 文件: {csv_file}")

# 读取 CSV
print("读取 CSV 内容:")
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# 清理临时文件
import os
if os.path.exists(csv_file):
    os.remove(csv_file)

# 4. asyncio - 异步I/O
# 现代 Python 高并发编程的核心
print("\n=== 4. asyncio - 异步I/O ===")
import asyncio
import time

# 定义一个异步函数
async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

# 运行异步任务
async def main():
    print(f"开始时间: {time.strftime('%X')}")
    
    # 并发运行两个任务
    # await say_after(1, 'hello') # 串行写法
    # await say_after(2, 'world')
    
    # 使用 gather 并发执行
    await asyncio.gather(
        say_after(1, 'hello'),
        say_after(2, 'world')
    )
    
    print(f"结束时间: {time.strftime('%X')}")

# 在 Jupyter/交互式环境中通常已有一个 loop，但在脚本中需要这样运行：
try:
    asyncio.run(main())
except RuntimeError:
    # 处理在某些特殊环境下 loop 已运行的情况
    pass

print("\n")
print("除了以上标准库，强烈推荐学习以下第三方库（需要 pip install）：")
print("1. requests: 处理 HTTP 请求，比标准库 urllib 好用太多")
print("2. pandas: 数据分析必备")
print("3. numpy: 科学计算基础")
print("4. flask/django: Web 开发框架")
