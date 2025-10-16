# 基本语法
# 1.单行注释
# print("#")

# 2.多行注释
# print('''#''')
# print("""#""")
'''
第三注释
第四注释
'''

"""
第五注释
第六注释
"""

#3.行与缩进
if 2 > 5:
    print("True")
else:
    print("\n\nFalse")

# 4. 多行语句
total = "item_one" +\
        "item_two" +\
        "item_three"

total = {"item_one","item_two","item_three"} # [],{},()中得多行语句，不需要 \\

# 5. 同一条显示多行语句
import sys; x = "\n你好！淼淼"; sys.stdout.write(x + "\n")

# 5.1 多个语句构成代码组
expression = input("Enter an expression: ")
if expression == "Hello" :
   print(1)
elif expression == "world" :
    print(2)
else :
   print(3)

# 6.字符串
str = "123456789"

print(str[2:])
print(str[1:5:2])
print(str * 2)
print(str + "你好呀")
print('hello\nworld')
print(r'hello\nrunoob')

input("\n\nPress the enter key to exit.")

import sys; x = "你好！"; sys.stdout.write(x + "\n")

number = int(input("Enter a number: "))

if number == 1:
    print("red")
elif number == 2:
    print("blue")
elif number == 3:
    print("pink")
else:
    print("Invalid number")

x = "A"
y = "B"
# 换行输出
print(x)
print(y)
print("------------------")
# 不换行输出
print(x, end=" ")
print(y, end=" ")
print("------------------")

# 7.空行 结果输出后 \n\n + 字符串
print("\n\n1234")

# import 与 from...import
print('================Python import mode==========================')
print ('命令行参数为:')
for i in sys.argv:
    print(i)
print ('\n python 路径为', sys.path)

from sys import argv, path  # 导入特定的成员

print('================python from import===================================')
print('path:', path)  # 因为已经导入path成员，所以此处引用时不需要加sys.path

# ========= 补充的Python基础知识 ==========

# 8. 变量和数据类型
print("\n========= 8. 变量和数据类型 ==========")
# 变量赋值
name = "Python"
age = 20
pi = 3.14159
is_valid = True

# 数据类型
print(f"name类型: {type(name)}")
print(f"age类型: {type(age)}")
print(f"pi类型: {type(pi)}")
print(f"is_valid类型: {type(is_valid)}")

# 类型转换
num_str = "123"
num_int = int(num_str)
print(f"字符串转整数: {num_int}, 类型: {type(num_int)}")

# 9. 基本运算符
print("\n========= 9. 基本运算符 ==========")
# 算术运算符
a, b = 10, 3
print(f"{a} + {b} = {a + b}")
print(f"{a} - {b} = {a - b}")
print(f"{a} * {b} = {a * b}")
print(f"{a} / {b} = {a / b}")  # 除法返回浮点数
print(f"{a} // {b} = {a // b}")  # 整除
print(f"{a} % {b} = {a % b}")  # 取余
print(f"{a} ** {b} = {a ** b}")  # 幂运算

# 比较运算符
print(f"{a} == {b}: {a == b}")
print(f"{a} != {b}: {a != b}")
print(f"{a} > {b}: {a > b}")
print(f"{a} < {b}: {a < b}")

# 逻辑运算符
x, y = True, False
print(f"{x} and {y}: {x and y}")
print(f"{x} or {y}: {x or y}")
print(f"not {x}: {not x}")

# 10. 列表（List）
print("\n========= 10. 列表（List） ==========")
# 创建列表
fruits = ["apple", "banana", "cherry", "orange"]
print(f"原始列表: {fruits}")

# 访问元素
print(f"第一个元素: {fruits[0]}")
print(f"最后一个元素: {fruits[-1]}")

# 修改元素
fruits[1] = "grape"
print(f"修改后的列表: {fruits}")

# 添加元素
fruits.append("pear")
print(f"添加元素后: {fruits}")
fruits.insert(2, "watermelon")
print(f"插入元素后: {fruits}")

# 删除元素
fruits.remove("cherry")
print(f"删除元素后: {fruits}")
popped = fruits.pop()
print(f"弹出最后元素: {popped}, 列表: {fruits}")

# 列表切片
print(f"切片[1:3]: {fruits[1:3]}")

# 列表长度
print(f"列表长度: {len(fruits)}")

# 11. 元组（Tuple）- 不可变序列
print("\n========= 11. 元组（Tuple） ==========")
tuple1 = (1, 2, 3, 4, 5)
tuple2 = "a", "b", "c"
print(f"元组1: {tuple1}")
print(f"元组2: {tuple2}")
print(f"访问元素: {tuple1[2]}")

# 12. 字典（Dictionary）
print("\n========= 12. 字典（Dictionary） ==========")
# 创建字典
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}
print(f"原始字典: {person}")

# 访问值
print(f"姓名: {person['name']}")
print(f"年龄: {person.get('age')}")

# 修改值
person['age'] = 26
print(f"修改后的字典: {person}")

# 添加键值对
person['job'] = "Engineer"
print(f"添加后的字典: {person}")

# 删除键值对
del person['city']
print(f"删除后的字典: {person}")

# 获取所有键和值
print(f"所有键: {list(person.keys())}")
print(f"所有值: {list(person.values())}")
print(f"所有键值对: {list(person.items())}")

# 13. 集合（Set）
print("\n========= 13. 集合（Set） ==========")
# 创建集合
colors = {"red", "green", "blue", "red", "yellow"}  # 自动去重
print(f"集合: {colors}")

# 添加元素
colors.add("purple")
print(f"添加后: {colors}")

# 删除元素
colors.remove("blue")
print(f"删除后: {colors}")

# 集合操作
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print(f"交集: {set1 & set2}")
print(f"并集: {set1 | set2}")
print(f"差集: {set1 - set2}")

# 14. 循环语句
print("\n========= 14. 循环语句 ==========")
# for循环
print("for循环遍历列表:")
for fruit in fruits:
    print(f"- {fruit}")

# for循环与range
print("\nfor循环与range:")
for i in range(1, 6):
    print(f"数字: {i}")

# while循环
print("\nwhile循环:")
count = 1
while count <= 5:
    print(f"计数: {count}")
    count += 1

# 15. 函数定义和调用
print("\n========= 15. 函数定义和调用 ==========")
# 定义函数
def greet(name):
    """这是一个问候函数"""
    return f"你好，{name}！"

# 调用函数
message = greet("Python学习者")
print(message)

# 带默认参数的函数
def calculate_area(width, height=10):
    return width * height

print(f"面积1: {calculate_area(5)}")  # 使用默认高度
print(f"面积2: {calculate_area(5, 20)}")  # 自定义高度

# 带可变参数的函数
def sum_numbers(*args):
    return sum(args)

print(f"求和: {sum_numbers(1, 2, 3, 4, 5)}")

# 16. 类和对象（面向对象编程基础）
print("\n========= 16. 类和对象 ==========")
# 定义类
class Dog:
    # 类变量
    species = "Canis familiaris"
    
    # 初始化方法
    def __init__(self, name, age):
        self.name = name  # 实例变量
        self.age = age
    
    # 实例方法
    def bark(self):
        return f"{self.name}说：汪汪！"

# 创建对象
my_dog = Dog("小黑", 3)
print(f"品种: {my_dog.species}")
print(f"名字: {my_dog.name}")
print(f"年龄: {my_dog.age}岁")
print(my_dog.bark())

# 17. 异常处理
print("\n========= 17. 异常处理 ==========")
try:
    result = 10 / 0
except ZeroDivisionError:
    print("错误: 除数不能为零！")
except Exception as e:
    print(f"发生错误: {e}")
else:
    print(f"结果: {result}")
finally:
    print("异常处理完成")

# 18. 文件操作
print("\n========= 18. 文件操作 ==========")
try:
    # 写入文件
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write("这是第一行\n")
        f.write("这是第二行\n")
    print("文件写入成功")
    
    # 读取文件
    with open("example.txt", "r", encoding="utf-8") as f:
        content = f.read()
    print(f"文件内容:\n{content}")
    
    # 逐行读取
    print("逐行读取:")
    with open("example.txt", "r", encoding="utf-8") as f:
        for line in f:
            print(f"- {line.strip()}")
            
    import os
    os.remove("example.txt")  # 删除示例文件
except Exception as e:
    print(f"文件操作出错: {e}")

# 19. 常用内置函数
print("\n========= 19. 常用内置函数 ==========")
# len() - 获取长度
print(f"列表长度: {len(fruits)}")

# max()/min() - 最大值/最小值
numbers = [3, 1, 4, 1, 5, 9]
print(f"最大值: {max(numbers)}")
print(f"最小值: {min(numbers)}")

# sorted() - 排序
print(f"排序后: {sorted(numbers)}")

# enumerate() - 枚举
print("枚举列表:")
for index, fruit in enumerate(fruits, 1):
    print(f"{index}. {fruit}")

# zip() - 打包
names = ["张三", "李四", "王五"]
scores = [85, 92, 78]
print("打包后的结果:")
for name, score in zip(names, scores):
    print(f"{name}: {score}分")

# map() - 映射
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(f"平方结果: {squared}")

# filter() - 过滤
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"偶数: {even_numbers}")

print("\n========= Python基础语法学习完成 ==========\n")

