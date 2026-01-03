# 数据可视化：Matplotlib 和 Seaborn
# 运行前请确保安装了库：pip install matplotlib seaborn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 设置支持中文显示 (Windows系统常用字体 SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

print("正在生成图表，请查看弹出的窗口...")

# 生成示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 1. Matplotlib 基础绘图
plt.figure(figsize=(10, 6)) # 设置画布大小

plt.plot(x, y1, label='Sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='Cos(x)', color='red', linestyle='--', linewidth=2)

plt.title('三角函数示例 (Matplotlib)')
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.legend() # 显示图例
plt.grid(True) # 显示网格

# 保存图片
# plt.savefig('plot_example.png')
# plt.show() # 在脚本中运行时，去掉注释以显示窗口

print("Matplotlib 绘图完成 (代码中已注释 show(), 可自行取消注释运行)")


# 2. Seaborn 高级绘图 (更美观，适合统计数据)
# 创建一个随机数据集
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 100),
    'Value': np.random.randn(100),
    'Score': np.random.randint(0, 100, 100)
})

plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Value', data=data)
plt.title('箱线图示例 (Seaborn)')
# plt.show()

print("Seaborn 绘图完成")

# 3. 散点图矩阵 (展示多个变量间的关系)
# 使用 seaborn 自带的 iris 数据集（如果网络通畅）
# 这里我们手动创建一个简单的数据集
df_pair = pd.DataFrame(np.random.randn(50, 4), columns=['A', 'B', 'C', 'D'])
df_pair['Label'] = np.random.choice(['X', 'Y'], 50)

sns.pairplot(df_pair, hue='Label')
# plt.show()
print("Pairplot 绘图完成")
