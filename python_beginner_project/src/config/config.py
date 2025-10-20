"""项目配置文件"""

# 应用配置
APP_NAME = "Python新手项目"
APP_VERSION = "0.1.0"

# 环境配置
class Config:
    """基础配置类"""
    DEBUG = False
    TESTING = False
    
class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    
class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    
class ProductionConfig(Config):
    """生产环境配置"""
    pass

# 根据环境选择配置
config_by_name = {
    "dev": DevelopmentConfig,
    "test": TestingConfig,
    "prod": ProductionConfig
}

# 默认配置
DEFAULT_CONFIG = config_by_name["dev"]