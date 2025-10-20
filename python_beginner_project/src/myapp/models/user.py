"""用户模型类"""

class User:
    """用户类，用于表示系统中的用户"""
    
    def __init__(self, username, email, age=None):
        """初始化用户对象
        
        Args:
            username: 用户名
            email: 电子邮件地址
            age: 年龄（可选）
        """
        self.username = username
        self.email = email
        self.age = age
    
    def get_info(self):
        """获取用户信息
        
        Returns:
            包含用户信息的字典
        """
        info = {
            "username": self.username,
            "email": self.email
        }
        if self.age is not None:
            info["age"] = self.age
        return info
    
    def is_adult(self):
        """检查用户是否为成年人
        
        Returns:
            如果年龄大于等于18岁返回True，否则返回False，未知年龄返回None
        """
        if self.age is None:
            return None
        return self.age >= 18