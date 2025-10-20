"""测试字符串工具函数"""

import unittest
from myapp.utils.string_utils import greet, format_message

class TestStringUtils(unittest.TestCase):
    """测试字符串工具函数的测试用例类"""
    
    def test_greet(self):
        """测试greet函数"""
        # 正常情况测试
        result = greet("张三")
        self.assertEqual(result, "你好，张三！欢迎学习Python！")
        
        # 边界情况测试
        result = greet("")
        self.assertEqual(result, "你好，！欢迎学习Python！")
    
    def test_format_message(self):
        """测试format_message函数"""
        # 正常情况测试
        template = "Hello, {name}! Welcome to {place}!"
        result = format_message(template, name="Alice", place="Python World")
        self.assertEqual(result, "Hello, Alice! Welcome to Python World!")
        
        # 多参数测试
        template = "{greeting}, {name}! You are {age} years old."
        result = format_message(template, greeting="Hi", name="Bob", age=25)
        self.assertEqual(result, "Hi, Bob! You are 25 years old.")

if __name__ == "__main__":
    unittest.main()