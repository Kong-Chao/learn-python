"""用户服务模块，处理用户相关的业务逻辑"""

from myapp.models.user import User

class UserService:
    """用户服务类"""
    
    def __init__(self):
        """初始化用户服务"""
        self.users = []  # 简单存储，实际项目中应使用数据库
    
    def create_user(self, username, email, age=None):
        """创建新用户
        
        Args:
            username: 用户名
            email: 电子邮件地址
            age: 年龄（可选）
        
        Returns:
            创建的用户对象
        """
        # 检查用户名是否已存在
        for user in self.users:
            if user.username == username:
                raise ValueError(f"用户名 '{username}' 已存在")
        
        # 创建新用户
        new_user = User(username, email, age)
        self.users.append(new_user)
        return new_user
    
    def get_user_by_username(self, username):
        """通过用户名获取用户
        
        Args:
            username: 要查找的用户名
        
        Returns:
            找到的用户对象，如果不存在返回None
        """
        for user in self.users:
            if user.username == username:
                return user
        return None
    
    def get_all_users(self):
        """获取所有用户
        
        Returns:
            用户对象列表
        """
        return self.users.copy()  # 返回副本，避免外部直接修改