"""项目使用示例"""

# 添加项目根目录到Python路径，使能正确导入myapp模块
import sys
sys.path.append('./src')

from myapp import greet, User
from myapp.api.user_api import UserAPI
from myapp.utils.string_utils import format_message

# 使用greet函数
print("=== 使用greet函数 ===")
greeting = greet("Python初学者")
print(greeting)
print()

# 使用format_message函数
print("=== 使用format_message函数 ===")
template = "{greeting}, {name}! 今天是学习Python的第{days}天。"
message = format_message(template, greeting="你好", name="小明", days=10)
print(message)
print()

# 使用User模型
print("=== 使用User模型 ===")
user = User("alice", "alice@example.com", 25)
user_info = user.get_info()
print(f"用户信息: {user_info}")
print(f"是否成年: {user.is_adult()}")
print()

# 使用UserAPI
print("=== 使用UserAPI ===")
user_api = UserAPI()

# 注册用户
print("注册用户:")
result, status_code = user_api.register_user("bob", "bob@example.com", 30)
print(f"结果: {result}")
print(f"状态码: {status_code}")
print()

# 尝试注册已存在的用户
print("尝试注册已存在的用户:")
result, status_code = user_api.register_user("bob", "bob_new@example.com", 31)
print(f"结果: {result}")
print(f"状态码: {status_code}")
print()

# 获取用户信息
print("获取用户信息:")
result, status_code = user_api.get_user("bob")
print(f"结果: {result}")
print(f"状态码: {status_code}")
print()

# 获取所有用户
print("获取所有用户:")
result, status_code = user_api.get_all_users()
print(f"结果: {result}")
print(f"状态码: {status_code}")