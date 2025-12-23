"""
AI分类模型训练示例：手写数字识别
这个例子演示如何训练一个AI模型来识别手写数字（0-9）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

print("=== 手写数字识别AI模型训练 ===")

# 1. 加载数据集
print("\n1. 加载手写数字数据集...")
digits = load_digits()

# 数据说明
print(f"数据集包含 {len(digits.images)} 个手写数字样本")
print(f"每个数字是 {digits.images[0].shape[0]}x{digits.images[0].shape[1]} 像素的图像")
print(f"数字类别: 0-9")

# 2. 数据预处理
print("\n2. 数据预处理...")

# 将图像数据展平为特征向量
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

print(f"特征数据形状: {X.shape}")
print(f"标签数据形状: {y.shape}")

# 3. 划分训练集和测试集
print("\n3. 划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]} 个样本")
print(f"测试集大小: {X_test.shape[0]} 个样本")

# 4. 创建和训练AI模型
print("\n4. 创建和训练随机森林分类器...")

# 使用随机森林算法
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

print("模型训练完成！")

# 5. 模型预测
print("\n5. 在测试集上进行预测...")
y_pred = model.predict(X_test)

# 6. 模型评估
print("\n6. 模型评估...")

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")

# 显示分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 7. 可视化一些预测结果
print("\n7. 可视化预测结果...")

# 选择一些测试样本进行可视化
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
fig.suptitle('手写数字识别结果 (预测/真实)', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(X_test):
        # 显示图像
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
        
        # 设置标题显示预测结果和真实标签
        ax.set_title(f'预测: {y_pred[i]}/真实: {y_test[i]}')
        
        # 如果预测正确，用绿色边框；错误用红色边框
        if y_pred[i] == y_test[i]:
            ax.spines['top'].set_color('green')
            ax.spines['bottom'].set_color('green')
            ax.spines['left'].set_color('green')
            ax.spines['right'].set_color('green')
            ax.spines['top'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
        else:
            ax.spines['top'].set_color('red')
            ax.spines['bottom'].set_color('red')
            ax.spines['left'].set_color('red')
            ax.spines['right'].set_color('red')
            ax.spines['top'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
        
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('models/digit_classification_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 混淆矩阵热力图
print("\n8. 生成混淆矩阵热力图...")

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵热力图')
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 特征重要性分析
print("\n9. 分析特征重要性...")

# 获取特征重要性
feature_importance = model.feature_importances_

# 将重要性映射回图像像素
importance_image = feature_importance.reshape(8, 8)

plt.figure(figsize=(8, 6))
plt.imshow(importance_image, cmap='hot', interpolation='nearest')
plt.colorbar(label='特征重要性')
plt.title('像素特征重要性热力图')
plt.xticks([])
plt.yticks([])
plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n特征重要性分析完成！颜色越亮表示该像素对分类越重要")

# 10. 保存模型
import joblib
joblib.dump(model, 'models/digit_classifier_model.pkl')
print("\n模型已保存到 models/digit_classifier_model.pkl")

# 11. 模型使用示例
print("\n=== 模型使用示例 ===")

# 加载保存的模型
loaded_model = joblib.load('models/digit_classifier_model.pkl')

# 对新数据进行预测（使用测试集中的一些样本）
print("\n对新数据进行预测:")
new_data_indices = [0, 5, 10]  # 测试集中的索引
for idx in new_data_indices:
    if idx < len(X_test):
        single_prediction = loaded_model.predict(X_test[idx].reshape(1, -1))
        print(f"样本 {idx}: 预测数字 = {single_prediction[0]}, 真实数字 = {y_test[idx]}")

print("\n=== 手写数字识别AI模型训练完成！ ===")
print("\n这个例子展示了：")
print("✓ 如何加载和处理图像数据")
print("✓ 如何使用随机森林算法训练分类模型") 
print("✓ 如何评估模型性能")
print("✓ 如何可视化和解释模型结果")
print("✓ 如何保存和加载训练好的模型")