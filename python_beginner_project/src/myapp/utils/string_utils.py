"""字符串处理工具函数"""

def greet(name):
    """向指定名称的人问好
    
    Args:
        name: 要问候的人的名称
    
    Returns:
        问候语字符串
    """
    return f"你好，{name}！欢迎学习Python！"

def format_message(message, **kwargs):
    """格式化消息字符串
    
    Args:
        message: 包含占位符的消息模板
        **kwargs: 用于替换占位符的键值对
    
    Returns:
        格式化后的消息字符串
    """
    return message.format(**kwargs)