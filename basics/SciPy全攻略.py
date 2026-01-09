# SciPy 全攻略：科学计算的核心力量
# 注意：运行此脚本需要安装 scipy 库
# 安装命令：pip install scipy

import sys
import numpy as np

try:
    import scipy
    from scipy import optimize, integrate, interpolate, signal, linalg, stats
except ImportError:
    print("错误：未找到 scipy 库。请运行 pip install scipy 安装")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("警告：未找到 matplotlib 库。部分绘图功能可能不可用。请运行 pip install matplotlib 安装")

print("========= SciPy 全攻略学习指南 ==========\n")
print(f"SciPy 版本: {scipy.__version__}\n")

# ==========================================
# 第一部分：SciPy 简介
# ==========================================
print("=== 1. SciPy 简介 ===")
print("SciPy 是建立在 NumPy 之上的高级科学计算库。")
print("它提供了大量数学算法和便利函数，用于解决科学和工程问题。")
print("核心模块包括：")
print("- optimize: 优化与寻根")
print("- integrate: 积分与微分方程求解")
print("- interpolate: 插值")
print("- signal: 信号处理")
print("- linalg: 线性代数 (比 NumPy 更强大)")
print("- stats: 统计分布与检验")
print("- special: 特殊函数")
print("\n")

# ==========================================
# 第二部分：最优化 (Optimization)
# ==========================================
print("=== 2. 最优化 (optimize) ===")

# 2.1 寻找函数最小值
print("--- 寻找函数最小值 ---")
# 定义一个函数 f(x) = x^2 + 10sin(x)
def f(x):
    return x**2 + 10*np.sin(x)

# 使用 BFGS 算法寻找最小值
# x0 是初始猜测值
result = optimize.minimize(f, x0=0)
print(f"最小化结果:\n{result}")
print(f"最小值点 x: {result.x}")
print(f"最小值 f(x): {result.fun}")

# 2.2 寻找方程的根 (Root Finding)
print("\n--- 寻找方程的根 ---")
# 求解方程 x^2 + 10sin(x) = 0
# 也就是寻找 f(x) = 0 的根
root = optimize.root(f, x0=1)
print(f"方程的根: {root.x}")

# 2.3 曲线拟合 (Curve Fitting)
print("\n--- 曲线拟合 ---")
# 生成带噪声的数据
x_data = np.linspace(-5, 5, num=50)
y_data = 2.9 * np.sin(1.5 * x_data) + np.random.normal(size=50)

# 定义拟合函数形式: y = a * sin(b * x)
def test_func(x, a, b):
    return a * np.sin(b * x)

# 进行拟合
params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
print(f"拟合参数: a={params[0]:.2f}, b={params[1]:.2f}")

# ==========================================
# 第三部分：积分 (Integration)
# ==========================================
print("\n=== 3. 积分 (integrate) ===")

# 3.1 定积分
print("--- 定积分 ---")
# 计算 f(x) = x^2 在 [0, 1] 区间的定积分
# 理论值: 1/3
result, error = integrate.quad(lambda x: x**2, 0, 1)
print(f"积分结果: {result}, 误差估计: {error}")

# 3.2 求解常微分方程 (ODE)
print("\n--- 求解常微分方程 (ODE) ---")
# 求解 dy/dt = -k * y, y(0) = 5
def model(y, t, k):
    dydt = -k * y
    return dydt

y0 = 5
t = np.linspace(0, 20)
k = 0.1
y = integrate.odeint(model, y0, t, args=(k,))
print(f"ODE 求解结果前5个点:\n{y[:5].flatten()}")

# ==========================================
# 第四部分：插值 (Interpolation)
# ==========================================
print("\n=== 4. 插值 (interpolate) ===")
x = np.linspace(0, 10, num=11)
y = np.cos(-x**2/9.0)

# 创建插值函数
f_interp = interpolate.interp1d(x, y, kind='cubic') # 三次样条插值

xnew = np.linspace(0, 10, num=41)
ynew = f_interp(xnew)
print(f"原始数据点数: {len(x)}, 插值后点数: {len(xnew)}")
print(f"插值结果前5个点: {ynew[:5]}")

# ==========================================
# 第五部分：线性代数 (Linear Algebra)
# ==========================================
print("\n=== 5. 线性代数 (linalg) ===")
# SciPy 的 linalg 比 NumPy 的 linalg 更快且功能更多
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

# 求解线性方程组 Ax = b
x = linalg.solve(A, b)
print(f"线性方程组解 x: {x}")

# 计算行列式
det = linalg.det(A)
print(f"行列式: {det}")

# 计算逆矩阵
inv = linalg.inv(A)
print(f"逆矩阵:\n{inv}")

# 特征值与特征向量
eigvals, eigvecs = linalg.eig(A)
print(f"特征值: {eigvals}")

# ==========================================
# 第六部分：信号处理 (Signal Processing)
# ==========================================
print("\n=== 6. 信号处理 (signal) ===")
# 6.1 卷积
sig = np.repeat([0., 1., 0.], 100)
win = signal.windows.hann(50)
filtered = signal.convolve(sig, win, mode='same') / sum(win)
print(f"卷积结果形状: {filtered.shape}")

# 6.2 寻找峰值
x = np.linspace(0, 10, 100)
y = np.sin(x)
peaks, _ = signal.find_peaks(y)
print(f"峰值索引: {peaks}, 峰值对应x: {x[peaks]}")

# ==========================================
# 第七部分：统计 (Statistics)
# ==========================================
print("\n=== 7. 统计 (stats) ===")
# 7.1 描述统计
data = np.random.normal(0, 1, 1000)
desc = stats.describe(data)
print(f"描述统计:\n{desc}")

# 7.2 概率分布
print("\n--- 正态分布 ---")
norm_dist = stats.norm(loc=0, scale=1) # 均值0，标准差1
print(f"PDF at x=0: {norm_dist.pdf(0)}") # 概率密度函数
print(f"CDF at x=0: {norm_dist.cdf(0)}") # 累积分布函数
print(f"随机采样 3 个数: {norm_dist.rvs(3)}")

# 7.3 假设检验 (T检验)
print("\n--- T检验 ---")
# 检验两组数据均值是否相同
rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
statistic, pvalue = stats.ttest_ind(rvs1, rvs2)
print(f"T统计量: {statistic:.4f}, P值: {pvalue:.4f}")
if pvalue > 0.05:
    print("结论: 无法拒绝原假设 (两组数据均值可能相同)")
else:
    print("结论: 拒绝原假设 (两组数据均值显著不同)")

print("\n========= 总结 ==========")
print("SciPy 是 Python 科学计算生态系统中极其重要的一环。")
print("它让 Python 具备了媲美 Matlab 的科学计算能力。")
print("本指南涵盖了最常用的功能，更多高级功能请查阅官方文档。")
