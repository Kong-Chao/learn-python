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
from tokenize import String

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

total = {"item_one","item_two","item_three"} # [],{},()中得多行语句，不需要 \

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

