"""
最简单的AI模型训练示例：线性回归
这个例子演示如何用Python训练一个AI模型来预测房价
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 准备训练数据
# 假设我们有一些房屋面积和价格的数据
print("=== 准备训练数据 ===")

# 房屋面积（平方米）
house_sizes = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140]).reshape(-1, 1)
# 对应的房价（万元）
house_prices = np.array([200, 240, 280, 320, 360, 400, 440, 480, 520, 560])

print(f"房屋面积数据: {house_sizes.flatten()}")
print(f"房屋价格数据: {house_prices}")

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    house_sizes, house_prices, test_size=0.2, random_state=42
)

print(f"\n训练集大小: {len(X_train)} 个样本")
print(f"测试集大小: {len(X_test)} 个样本")

# 3. 创建和训练AI模型
print("\n=== 创建和训练AI模型 ===")

# 使用线性回归模型
model = LinearRegression()

# 训练模型（这就是AI学习的过程）
model.fit(X_train, y_train)

print("模型训练完成！")
print(f"模型系数 (斜率): {model.coef_[0]:.2f}")
print(f"模型截距: {model.intercept_:.2f}")

# 4. 模型预测
print("\n=== 模型预测 ===")

# 在测试集上进行预测
y_pred = model.predict(X_test)

print(f"真实价格: {y_test}")
print(f"预测价格: {y_pred.round(2)}")

# 5. 模型评估
print("\n=== 模型评估 ===")

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
# 计算R²分数（决定系数）
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.2f}")
print(f"R²分数: {r2:.3f}")

# 6. 可视化结果
print("\n=== 可视化结果 ===")

plt.figure(figsize=(10, 6))

# 绘制训练数据点
plt.scatter(X_train, y_train, color='blue', label='训练数据', alpha=0.7)

# 绘制测试数据点
plt.scatter(X_test, y_test, color='red', label='测试数据', alpha=0.7)

# 绘制预测线
x_line = np.linspace(40, 150, 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='green', linewidth=2, label='预测模型')

plt.xlabel('房屋面积 (平方米)')
plt.ylabel('房屋价格 (万元)')
plt.title('AI模型预测：房屋面积 vs 价格')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('models/house_price_prediction.png')
plt.show()

# 7. 使用模型进行新预测
print("\n=== 新数据预测 ===")

new_sizes = np.array([[65], [95], [125]])  # 新的房屋面积
new_predictions = model.predict(new_sizes)

for i, size in enumerate(new_sizes):
    print(f"房屋面积 {size[0]} 平方米 -> 预测价格: {new_predictions[i]:.2f} 万元")

# 8. 保存模型
import joblib
joblib.dump(model, 'models/house_price_model.pkl')
print("\n模型已保存到 models/house_price_model.pkl")

print("\n=== AI模型训练完成！ ===")
print("这个简单的例子展示了AI模型训练的基本流程：")
print("1. 准备数据 -> 2. 选择模型 -> 3. 训练模型 -> 4. 评估模型 -> 5. 使用模型")