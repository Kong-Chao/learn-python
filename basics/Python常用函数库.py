# Python新手必备函数库指南
print("========= Python常用函数库 ==========\n")

"""
Python的强大之处在于它拥有丰富的函数库。确实如你所说，Python编程很大程度上就是了解
并合理调用各种库来实现功能。下面介绍Python新手必须掌握的常用函数库。
"""

# 第一部分：Python标准库（无需安装，Python自带）
print("\n=== 第一部分：Python标准库 ===\n")

# 1. os - 操作系统接口
print("1. os - 操作系统接口")
print("   用途：提供与操作系统交互的功能")
print("   常用功能：文件操作、目录管理、环境变量等")
print("   示例：")
'''
import os

# 获取当前目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 创建目录
# os.makedirs("test_dir", exist_ok=True)

# 列出目录内容
files = os.listdir(current_dir)
print(f"目录内容: {files[:5]}")  # 只显示前5个
'''
print("\n")

# 2. sys - Python解释器相关
print("2. sys - Python解释器相关")
print("   用途：提供与Python解释器交互的功能")
print("   常用功能：命令行参数、Python路径、退出程序等")
print("   示例：")
'''
import sys

# 获取Python版本
print(f"Python版本: {sys.version}")

# 获取命令行参数
print(f"命令行参数: {sys.argv}")

# 添加模块搜索路径
sys.path.append("./src")
'''
print("\n")

# 3. datetime - 日期和时间处理
print("3. datetime - 日期和时间处理")
print("   用途：处理日期和时间")
print("   常用功能：获取当前时间、日期计算、格式化等")
print("   示例：")
'''
from datetime import datetime, timedelta

# 获取当前时间
now = datetime.now()
print(f"当前时间: {now}")

# 格式化时间
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"格式化时间: {formatted}")

# 日期计算
tomorrow = now + timedelta(days=1)
print(f"明天: {tomorrow.strftime('%Y-%m-%d')}")
'''
print("\n")

# 4. json - JSON数据处理
print("4. json - JSON数据处理")
print("   用途：处理JSON格式数据")
print("   常用功能：JSON字符串和Python对象的相互转换")
print("   示例：")
'''
import json

# Python对象转JSON
user_dict = {"name": "张三", "age": 25, "city": "北京"}
json_str = json.dumps(user_dict, ensure_ascii=False)
print(f"JSON字符串: {json_str}")

# JSON字符串转Python对象
json_data = '{"name": "李四", "age": 30}'
data = json.loads(json_data)
print(f"Python对象: {data}")
print(f"姓名: {data['name']}")
'''
print("\n")

# 5. re - 正则表达式
print("5. re - 正则表达式")
print("   用途：字符串匹配和处理")
print("   常用功能：模式匹配、替换、提取等")
print("   示例：")
'''
import re

# 匹配邮箱
email = "test@example.com"
pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
if re.match(pattern, email):
    print(f"{email} 是有效的邮箱格式")

# 提取数字
text = "价格是 123.45 元"
nums = re.findall(r'\d+\.?\d*', text)
print(f"提取的数字: {nums}")
'''
print("\n")

# 6. random - 随机数生成
print("6. random - 随机数生成")
print("   用途：生成随机数和随机选择")
print("   常用功能：随机数、随机选择、打乱序列等")
print("   示例：")
'''
import random

# 生成随机整数
rand_int = random.randint(1, 100)
print(f"随机整数 (1-100): {rand_int}")

# 随机选择
fruits = ["苹果", "香蕉", "橙子", "葡萄"]
selected = random.choice(fruits)
print(f"随机选择的水果: {selected}")

# 打乱序列
random.shuffle(fruits)
print(f"打乱后的水果列表: {fruits}")
'''
print("\n")

# 7. math - 数学计算
print("7. math - 数学计算")
print("   用途：提供数学函数")
print("   常用功能：三角函数、指数、对数、取整等")
print("   示例：")
'''
import math

# 基本运算
print(f"π: {math.pi}")
print(f"e: {math.e}")
print(f"平方根(16): {math.sqrt(16)}")

# 三角函数
print(f"sin(π/2): {math.sin(math.pi/2)}")

# 取整
print(f"向上取整(4.2): {math.ceil(4.2)}")
print(f"向下取整(4.9): {math.floor(4.9)}")
'''
print("\n")

# 8. collections - 高级数据结构
print("8. collections - 高级数据结构")
print("   用途：提供额外的数据类型")
print("   常用功能：计数器、有序字典、默认字典等")
print("   示例：")
'''
from collections import Counter, defaultdict, OrderedDict

# 计数器
text = "hello world hello python"
word_count = Counter(text.split())
print(f"单词计数: {dict(word_count)}")

# 默认字典
d = defaultdict(int)
d["a"] += 1
d["b"] += 2
print(f"默认字典: {dict(d)}")

# 有序字典（Python 3.7+ 普通字典也保持插入顺序）
od = OrderedDict()
od["c"] = 3
od["a"] = 1
od["b"] = 2
print(f"有序字典: {dict(od)}")
'''
print("\n")

# 9. time - 时间相关函数
print("9. time - 时间相关函数")
print("   用途：处理时间延迟和计时")
print("   常用功能：延时、时间戳、性能计时等")
print("   示例：")
'''
import time

# 获取时间戳
print(f"当前时间戳: {time.time()}")

# 延时
# print("等待2秒...")
# time.sleep(2)
# print("继续执行")

# 性能计时
start_time = time.time()
# 执行一些操作
sum(range(1000000))
end_time = time.time()
print(f"执行时间: {end_time - start_time:.6f} 秒")
'''
print("\n")

# 10. logging - 日志记录
print("10. logging - 日志记录")
print("   用途：记录程序运行日志")
print("   常用功能：不同级别的日志、文件日志、格式化等")
print("   示例：")
'''
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 记录不同级别的日志
logger.debug("调试信息")  # 不会显示，因为级别设置为INFO
logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
logger.critical("严重错误日志")
'''
print("\n")

# 第二部分：常用第三方库（需要单独安装）
print("\n=== 第二部分：常用第三方库 ===\n")
print("注意：第三方库需要使用pip安装，例如：pip install numpy\n")

# 1. NumPy - 数值计算
print("1. NumPy - 数值计算库")
print("   用途：科学计算、数组操作、线性代数等")
print("   特点：高性能、多维数组支持、丰富的数学函数")
print("   安装：pip install numpy")
print("   示例：")
'''
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy数组: {arr}")
print(f"数组形状: {arr.shape}")

# 数组运算
print(f"数组+1: {arr + 1}")
print(f"数组平方: {arr ** 2}")

# 矩阵操作
matrix = np.array([[1, 2], [3, 4]])
print(f"矩阵:\n{matrix}")
print(f"矩阵转置:\n{matrix.T}")
'''
print("\n")

# 2. Pandas - 数据分析
print("2. Pandas - 数据分析库")
print("   用途：数据处理、数据分析、数据可视化基础")
print("   特点：强大的数据结构、数据清洗、合并、分组等功能")
print("   安装：pip install pandas")
print("   示例：")
'''
import pandas as pd

# 创建DataFrame
data = {
    'Name': ['张三', '李四', '王五'],
    'Age': [25, 30, 35],
    'City': ['北京', '上海', '广州']
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)

# 基本操作
print("\n前2行:")
print(df.head(2))

print("\n年龄统计:")
print(df['Age'].describe())
'''
print("\n")

# 3. Matplotlib - 数据可视化
print("3. Matplotlib - 数据可视化库")
print("   用途：创建图表、图形和数据可视化")
print("   特点：支持多种图表类型、高度可定制")
print("   安装：pip install matplotlib")
print("   示例：")
'''
import matplotlib.pyplot as plt

# 创建简单图表
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', color='blue', label='示例数据')
plt.title('简单折线图')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.legend()
plt.grid(True)
# plt.show()  # 显示图表
'''
print("\n")

# 4. Requests - HTTP请求
print("4. Requests - HTTP请求库")
print("   用途：发送HTTP请求，获取网页内容或API数据")
print("   特点：简单易用、功能强大、支持各种HTTP方法")
print("   安装：pip install requests")
print("   示例：")
'''
import requests

# 发送GET请求
# response = requests.get("https://httpbin.org/get")
# print(f"状态码: {response.status_code}")
# print(f"响应内容: {response.json()}")

# 发送POST请求
# data = {"name": "张三", "age": 25}
# response = requests.post("https://httpbin.org/post", json=data)
# print(f"POST响应: {response.json()}")
'''
print("\n")

# 5. BeautifulSoup - 网页解析
print("5. BeautifulSoup - 网页解析库")
print("   用途：解析HTML和XML文档")
print("   特点：提取网页数据、HTML元素定位、数据清洗")
print("   安装：pip install beautifulsoup4")
print("   示例：")
'''
from bs4 import BeautifulSoup
import requests

# 示例HTML解析
html_doc = """
<html>
<body>
    <h1>示例网页</h1>
    <p class="content">这是一个段落。</p>
    <ul>
        <li>项目1</li>
        <li>项目2</li>
    </ul>
</body>
</html>
"""

soup = BeautifulSoup(html_doc, 'html.parser')

# 提取元素
print(f"标题: {soup.h1.text}")
print(f"段落内容: {soup.find('p', class_='content').text}")

# 提取列表项
print("列表项:")
for li in soup.find_all('li'):
    print(f"- {li.text}")
'''
print("\n")

# 6. Flask - Web开发
print("6. Flask - Web开发框架")
print("   用途：创建Web应用和API")
print("   特点：轻量级、灵活、易于学习")
print("   安装：pip install flask")
print("   示例：")
'''
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Flask!"

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "这是API数据", "status": "success"})

if __name__ == '__main__':
    # app.run(debug=True)  # 运行Flask应用
    pass
'''
print("\n")

# 7. SQLAlchemy - 数据库ORM
print("7. SQLAlchemy - 数据库ORM")
print("   用途：数据库操作和对象关系映射")
print("   特点：支持多种数据库、面向对象的数据库操作")
print("   安装：pip install sqlalchemy")
print("   示例：")
'''
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 创建数据库连接
engine = create_engine('sqlite:///:memory:')  # 内存数据库
Base = declarative_base()
Session = sessionmaker(bind=engine)

# 定义模型
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

# 创建表
# Base.metadata.create_all(engine)
'''
print("\n")

# 8. Pytest - 测试框架
print("8. Pytest - 测试框架")
print("   用途：编写和运行测试")
print("   特点：简单、强大、支持参数化测试等高级功能")
print("   安装：pip install pytest")
print("   示例：")
'''
def add(a, b):
    return a + b

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
'''
print("\n")

# 9. Pillow - 图像处理
print("9. Pillow - 图像处理库")
print("   用途：图像处理、编辑、转换等")
print("   特点：支持多种图像格式、基本图像操作")
print("   安装：pip install pillow")
print("   示例：")
'''
from PIL import Image, ImageFilter

# # 打开图像
# img = Image.open('example.jpg')

# # 图像操作
# gray_img = img.convert('L')  # 转为灰度图
# resized_img = img.resize((300, 200))  # 调整大小
# blurred_img = img.filter(ImageFilter.BLUR)  # 模糊处理

# # 保存图像
# # gray_img.save('gray_example.jpg')
'''
print("\n")

# 10. OpenCV - 计算机视觉
print("10. OpenCV - 计算机视觉库")
print("   用途：图像处理、计算机视觉、机器学习")
print("   特点：高性能、丰富的计算机视觉算法")
print("   安装：pip install opencv-python")
print("   示例：")
'''
import cv2

# # 读取图像
# img = cv2.imread('example.jpg')

# # 图像处理
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
# edges = cv2.Canny(gray, 100, 200)  # 边缘检测

# # 显示图像
# # cv2.imshow('Original', img)
# # cv2.imshow('Gray', gray)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
'''
print("\n")

# 第三部分：Python项目开发建议
print("\n=== 第三部分：Python项目开发建议 ===\n")

print("1. 库的选择原则:")
print("   - 优先使用标准库，其次是成熟的第三方库")
print("   - 选择文档完善、社区活跃的库")
print("   - 关注库的维护状态和兼容性\n")

print("2. 环境管理:")
print("   - 使用虚拟环境（virtualenv、conda）管理项目依赖")
print("   - 使用requirements.txt记录依赖版本")
print("   - 示例：pip freeze > requirements.txt\n")

print("3. 学习资源:")
print("   - 官方文档是最好的学习资源")
print("   - Python官方文档: https://docs.python.org/")
print("   - 各个库的官方文档")
print("   - GitHub上的示例代码\n")

print("4. 实践建议:")
print("   - 从小项目开始，逐步使用更多的库")
print("   - 阅读优秀项目的源码，学习库的使用方式")
print("   - 参与开源项目，积累实战经验\n")

print("总结：Python的生态系统非常丰富，掌握常用库的使用对于提高编程效率至关重要。")
print("学习过程中，建议先掌握标准库，再根据需要学习特定领域的第三方库。")
print("记住：好的Python程序员不是记住所有库的用法，而是知道在什么场景下使用什么库，")
print("并能够快速查阅文档解决问题。\n")