# 机器学习入门：Scikit-learn (sklearn)
# 运行前请确保安装了库：pip install scikit-learn

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("========= 机器学习初体验：线性回归 ==========\n")

# 1. 准备数据
# 假设我们要预测房屋价格：面积 (X) -> 价格 (y)
# 生成一些模拟数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1) # 100个样本，1个特征（面积）
y = 4 + 3 * X + np.random.randn(100, 1) # 真实关系：y = 4 + 3x + 噪声

print(f"数据形状: X={X.shape}, y={y.shape}")
print("前5个样本:\n", np.hstack((X[:5], y[:5])))

# 2. 划分训练集和测试集
# 80% 用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

# 3. 选择模型并训练
model = LinearRegression() # 线性回归模型
print("\n开始训练模型...")
model.fit(X_train, y_train)

print("模型训练完成！")
print(f"截距 (Intercept): {model.intercept_}")
print(f"系数 (Coefficient): {model.coef_}")
print(f"学到的公式近似为: y = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f} * x")

# 4. 预测与评估
y_pred = model.predict(X_test)

# 评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估:")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"R2 分数 (越接近1越好): {r2:.2f}")

# 5. 简单预测
new_area = np.array([[1.5]]) # 预测面积为 1.5 的房子
price = model.predict(new_area)
print(f"\n预测面积为 1.5 的房子价格: {price[0][0]:.2f}")

print("\n\n========= 其他常见机器学习算法 ==========")
print("1. 分类 (Classification): 预测类别 (如邮件是否垃圾邮件)")
print("   - 逻辑回归 (LogisticRegression)")
print("   - 支持向量机 (SVM)")
print("   - 决策树 (DecisionTree)")
print("   - 随机森林 (RandomForest)")
print("\n2. 聚类 (Clustering): 无监督分组")
print("   - K-Means")
print("\n3. 降维 (Dimensionality Reduction)")
print("   - PCA (主成分分析)")
