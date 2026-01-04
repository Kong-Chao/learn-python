print("Hello, Python!")

# python 的日志一般如何引入使用？举例说明并测试运行看看
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 记录日志
logging.info("这是一条INFO日志")
logging.warning("这是一条WARNING日志")
logging.error("这是一条ERROR日志")