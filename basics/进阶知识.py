# Python进阶知识学习指南
print("========= Python进阶知识学习 ==========\n")

# 1. 装饰器（Decorators）
print("\n========= 1. 装饰器（Decorators） ==========")

# 基本装饰器示例
def my_decorator(func):
    def wrapper():
        print("装饰器执行前")
        func()
        print("装饰器执行后")
    return wrapper

@my_decorator
def say_hello():
    print("Hello, 装饰器!")

# 使用装饰器
say_hello()

# 带参数的装饰器
def with_params(decorator_param):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"装饰器参数: {decorator_param}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@with_params("Python进阶")
def greet(name):
    return f"你好, {name}!"

print(greet("学习者"))

# 2. 生成器（Generators）和迭代器（Iterators）
print("\n========= 2. 生成器和迭代器 ==========")

# 生成器函数
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1

print("生成器示例:")
for num in count_up_to(5):
    print(num, end=" ")
print()

# 迭代器示例
class MyIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        value = self.data[self.index]
        self.index += 1
        return value

print("迭代器示例:")
my_iter = MyIterator(["Python", "Java", "C++"])
for item in my_iter:
    print(item, end=" ")
print()

# 3. 上下文管理器（Context Managers）
print("\n========= 3. 上下文管理器 ==========")

# 使用with语句自动管理资源
with open("temp_file.txt", "w", encoding="utf-8") as f:
    f.write("这是通过with语句创建的文件")

with open("temp_file.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(f"文件内容: {content}")

# 自定义上下文管理器
class MyContextManager:
    def __enter__(self):
        print("进入上下文")
        return "上下文对象"
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("退出上下文")
        # 返回True表示异常已处理
        return False

print("自定义上下文管理器:")
with MyContextManager() as cm:
    print(f"使用{cm}")

# 使用contextlib简化上下文管理器
from contextlib import contextmanager

@contextmanager
def simple_context():
    print("开始")
    yield "简化的上下文"
    print("结束")

print("简化的上下文管理器:")
with simple_context() as sc:
    print(f"使用{sc}")

import os
os.remove("temp_file.txt")  # 清理临时文件

# 4. 多线程和多进程
print("\n========= 4. 多线程和多进程 ==========")

# 多线程示例
import threading
import time

def thread_task(name):
    print(f"线程 {name} 开始")
    time.sleep(2)
    print(f"线程 {name} 结束")

threads = []
for i in range(3):
    t = threading.Thread(target=thread_task, args=(f"T-{i}",))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("所有线程执行完成")

# 多进程示例 - 注意：在某些环境中可能需要特殊处理
print("\n多进程示例:")
try:
    from multiprocessing import Process
    
    def process_task(name):
        print(f"进程 {name} 运行")
    
    processes = []
    for i in range(2):
        p = Process(target=process_task, args=(f"P-{i}",))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("所有进程执行完成")
except Exception as e:
    print(f"多进程执行需注意环境限制: {e}")

# 5. 异常处理进阶
print("\n========= 5. 异常处理进阶 ==========")

# 自定义异常
class CustomError(Exception):
    """自定义异常类"""
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super().__init__(self.message)

# 异常链
print("异常链示例:")
try:
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        # 抛出新异常并保留原始异常上下文
        raise CustomError("计算错误", 500) from e
except CustomError as e:
    print(f"捕获自定义异常: {e}, 错误码: {e.code}")
    print(f"原始异常: {e.__cause__}")

# 6. 正则表达式（Regular Expressions）
print("\n========= 6. 正则表达式 ==========")

import re

# 基本匹配
pattern = r'\d+'  # 匹配一个或多个数字
text = "我的电话号码是1234567890，还有9876543210"
matches = re.findall(pattern, text)
print(f"找到的数字: {matches}")

# 分组匹配
pattern = r'(\d{3})-(\d{4})-(\d{4})'  # 匹配电话号码格式
phone_text = "联系电话：123-4567-8901 和 987-6543-2109"
match = re.search(pattern, phone_text)
if match:
    print(f"完整号码: {match.group(0)}")
    print(f"区号: {match.group(1)}")
    print(f"中间部分: {match.group(2)}")
    print(f"最后部分: {match.group(3)}")

# 替换文本
pattern = r'bad|evil'  # 匹配多个关键词
text = "这个主意太bad了，简直是evil！"
new_text = re.sub(pattern, "good", text)
print(f"替换后: {new_text}")

# 7. 模块和包管理
print("\n========= 7. 模块和包管理 ==========")

# 创建临时模块示例（仅演示代码结构）
print("模块导入示例:")
# 导入已存在的模块
import math
print(f"π的值: {math.pi}")
print(f"sin(π/2): {math.sin(math.pi/2)}")

# 相对导入说明
print("\n包和相对导入:")
print("# 在实际项目中，包结构示例:")
print("""
my_package/            # 包目录
├── __init__.py        # 初始化文件，使其成为包
├── module1.py         # 模块1
└── subpackage/        # 子包
    ├── __init__.py
    └── module2.py
""")
print("# 相对导入示例:")
print("from . import module1     # 导入同级模块")
print("from ..subpackage import module2  # 导入子包模块")

# 8. 面向对象编程进阶
print("\n========= 8. 面向对象编程进阶 ==========")

# 继承
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("子类必须实现此方法")

class Dog(Animal):
    def speak(self):
        return f"{self.name} 说: 汪汪!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} 说: 喵喵!"

# 多态
animals = [Dog("小黑"), Cat("小花")]
print("多态示例:")
for animal in animals:
    print(animal.speak())

# 抽象类和接口
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

rect = Rectangle(4, 5)
print(f"矩形面积: {rect.area()}")
print(f"矩形周长: {rect.perimeter()}")

# 9. 文件和目录操作
print("\n========= 9. 文件和目录操作 ==========")

import os
import shutil

# 创建目录
print("目录操作示例:")
temp_dir = "temp_folder"
os.makedirs(temp_dir, exist_ok=True)
print(f"创建目录: {temp_dir}")

# 列出目录内容
print(f"当前目录内容: {os.listdir('.')}")

# 创建文件
with open(os.path.join(temp_dir, "test.txt"), "w") as f:
    f.write("测试文件内容")

# 检查文件是否存在
file_path = os.path.join(temp_dir, "test.txt")
print(f"文件存在? {os.path.exists(file_path)}")
print(f"是文件? {os.path.isfile(file_path)}")
print(f"是目录? {os.path.isdir(temp_dir)}")

# 获取文件信息
print(f"文件大小: {os.path.getsize(file_path)} 字节")
print(f"绝对路径: {os.path.abspath(file_path)}")

# 删除文件和目录
os.remove(file_path)
shutil.rmtree(temp_dir)
print(f"已删除临时目录: {temp_dir}")

# 10. 数据库操作
print("\n========= 10. 数据库操作 ==========")

# SQLite数据库示例
import sqlite3

try:
    # 连接数据库（如果不存在则创建）
    conn = sqlite3.connect(':memory:')  # 使用内存数据库
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')
    
    # 插入数据
    users = [(1, '张三', 25), (2, '李四', 30), (3, '王五', 35)]
    cursor.executemany('INSERT INTO users VALUES (?, ?, ?)', users)
    conn.commit()
    
    # 查询数据
    cursor.execute('SELECT * FROM users WHERE age > 28')
    print("查询结果:")
    for row in cursor.fetchall():
        print(row)
    
    # 关闭连接
    conn.close()
except Exception as e:
    print(f"数据库操作错误: {e}")

# 11. 网络编程基础
print("\n========= 11. 网络编程基础 ==========")

# 简单的HTTP请求示例
import urllib.request

try:
    print("发送HTTP请求示例:")
    # 注意：在实际环境中可能需要处理代理和证书问题
    # response = urllib.request.urlopen('https://httpbin.org/get')
    # data = response.read()
    # print(f"响应状态: {response.status}")
    print("由于环境限制，HTTP请求示例可能无法直接运行")
    print("实际使用示例:")
    print("import requests")
    print("response = requests.get('https://httpbin.org/get')")
    print("print(response.json())")
except Exception as e:
    print(f"网络请求错误: {e}")

# 12. 常用设计模式
print("\n========= 12. 常用设计模式 ==========")

# 单例模式
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
            cls._instance.data = "单例数据"
        return cls._instance

# 工厂模式
class ShapeFactory:
    @staticmethod
    def create_shape(shape_type):
        if shape_type == 'circle':
            return Circle()
        elif shape_type == 'rectangle':
            return Rectangle(1, 1)
        else:
            raise ValueError(f"未知的形状类型: {shape_type}")

class Circle:
    def draw(self):
        return "绘制圆形"

# 测试单例模式
s1 = Singleton()
s2 = Singleton()
print(f"单例模式 - s1和s2是同一个对象? {s1 is s2}")

# 测试工厂模式
factory = ShapeFactory()
circle = factory.create_shape('circle')
print(f"工厂模式 - {circle.draw()}")

# 13. 性能优化
print("\n========= 13. 性能优化 ==========")

# 使用列表推导式（比循环更高效）
print("列表推导式 vs 普通循环:")
import time

# 普通循环
start_time = time.time()
squares1 = []
for i in range(100000):
    squares1.append(i ** 2)
loop_time = time.time() - start_time

# 列表推导式
start_time = time.time()
squares2 = [i ** 2 for i in range(100000)]
comprehension_time = time.time() - start_time

print(f"普通循环时间: {loop_time:.6f}秒")
print(f"列表推导式时间: {comprehension_time:.6f}秒")
print(f"列表推导式提升: {loop_time/comprehension_time:.2f}倍")

# 使用生成器节省内存
print("\n生成器 vs 列表:")
def large_list():
    return [i for i in range(1000000)]

def large_generator():
    for i in range(1000000):
        yield i

import sys
list_obj = large_list()
gen_obj = large_generator()
print(f"列表占用内存: {sys.getsizeof(list_obj)} 字节")
print(f"生成器占用内存: {sys.getsizeof(gen_obj)} 字节")

# 14. 单元测试
print("\n========= 14. 单元测试 ==========")

import unittest

# 待测试的函数
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# 测试类
class TestMathFunctions(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(-1, -1), -2)
    
    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)
        self.assertEqual(subtract(0, 0), 0)
        self.assertEqual(subtract(-1, -1), 0)

# 运行测试
print("运行简单测试:")
# 创建测试套件
suite = unittest.TestLoader().loadTestsFromTestCase(TestMathFunctions)
# 运行测试
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print("\n========= Python进阶知识学习指南完成 ==========\n")
print("建议学习路径:")
print("1. 先掌握装饰器、生成器和上下文管理器")
print("2. 学习面向对象编程进阶概念")
print("3. 掌握文件操作和异常处理进阶")
print("4. 学习正则表达式和数据库操作")
print("5. 了解多线程/多进程编程")
print("6. 最后学习设计模式和性能优化")