"""myapp包初始化文件"""

__version__ = "0.1.0"
__author__ = "Python初学者"

# 从子模块导入常用功能
from myapp.utils.string_utils import greet
from myapp.models.user import User

__all__ = ["greet", "User"]