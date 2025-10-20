"""用户相关的API接口"""

from myapp.services.user_service import UserService

class UserAPI:
    """用户API类，提供用户相关的接口"""
    
    def __init__(self):
        """初始化API，创建服务实例"""
        self.user_service = UserService()
    
    def register_user(self, username, email, age=None):
        """注册新用户API
        
        Args:
            username: 用户名
            email: 电子邮件地址
            age: 年龄（可选）
        
        Returns:
            包含用户信息的字典和状态码
        """
        try:
            user = self.user_service.create_user(username, email, age)
            return {
                "status": "success",
                "data": user.get_info()
            }, 201  # 201表示资源创建成功
        except ValueError as e:
            return {
                "status": "error",
                "message": str(e)
            }, 400  # 400表示请求错误
    
    def get_user(self, username):
        """获取用户信息API
        
        Args:
            username: 用户名
        
        Returns:
            包含用户信息的字典和状态码
        """
        user = self.user_service.get_user_by_username(username)
        if user:
            return {
                "status": "success",
                "data": user.get_info()
            }, 200  # 200表示成功
        else:
            return {
                "status": "error",
                "message": f"用户 '{username}' 不存在"
            }, 404  # 404表示资源未找到
    
    def get_all_users(self):
        """获取所有用户API
        
        Returns:
            包含所有用户信息的列表和状态码
        """
        users = self.user_service.get_all_users()
        return {
            "status": "success",
            "data": [user.get_info() for user in users],
            "total": len(users)
        }, 200