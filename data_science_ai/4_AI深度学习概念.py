# 深度学习与AI概念：PyTorch / TensorFlow
# 注意：深度学习库通常较大，且需要配置环境（如 CUDA 加速），这里仅作为概念展示。
# 推荐使用 PyTorch，它是目前学术界和工业界最流行的深度学习框架之一。

print("========= 深度学习 (Deep Learning) 概念 ==========\n")

print("""
深度学习是机器学习的一个子集，主要使用"神经网络"来解决复杂问题。
核心库：
1. PyTorch (Facebook推出): 灵活，Pythonic，目前最受欢迎。
2. TensorFlow (Google推出): 工业部署能力强，老牌框架。
""")

# 下面是 PyTorch 的伪代码/示例代码
# 如果你安装了 torch (pip install torch)，可以取消注释运行

'''
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 张量 (Tensor) - 深度学习的基本数据单元 (类似于 NumPy 的 Array，但可以在 GPU 上运行)
x = torch.tensor([[1., 2.], [3., 4.]])
print(f"PyTorch 张量:\n{x}")

# 2. 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 定义层：输入维度 10 -> 隐藏层 5 -> 输出维度 1
        self.layer1 = nn.Linear(10, 5)
        self.relu = nn.ReLU() # 激活函数
        self.layer2 = nn.Linear(5, 1)
    
    def forward(self, x):
        # 前向传播过程
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 3. 初始化模型
model = SimpleNet()
print("\n模型结构:")
print(model)

# 4. 定义损失函数和优化器
criterion = nn.MSELoss() # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01) # 随机梯度下降

# 5. 模拟训练步骤
# inputs = torch.randn(1, 10) # 随机输入
# target = torch.randn(1, 1)  # 随机目标
# 
# optimizer.zero_grad()   # 清空梯度
# output = model(inputs)  # 前向传播
# loss = criterion(output, target) # 计算损失
# loss.backward()         # 反向传播 (计算梯度)
# optimizer.step()        # 更新参数
# 
# print(f"训练一次后的 Loss: {loss.item()}")
'''

print("\n========= 如何开始 AI 学习 ==========")
print("1. 数学基础: 线性代数 (矩阵运算), 概率论, 微积分 (梯度下降原理)")
print("2. 编程基础: Python, NumPy, Pandas (你已经在学了)")
print("3. 机器学习基础: 理解什么是回归、分类、过拟合、欠拟合 (Scikit-learn)")
print("4. 深度学习: 学习神经网络原理，CNN (图像), RNN/Transformer (文本/NLP)")
print("5. 实战: 参加 Kaggle 比赛，复现经典论文")
