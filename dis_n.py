import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
# np.random.seed(24)

# 创建画布
fig, ax = plt.subplots(figsize=(8, 5))
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig2, ax2 = plt.subplots(figsize=(8, 5))

# 定义离散的 x 轴范围，并将其归一化到 [0, 1]
x = np.linspace(0, 1, 100)  # 100 个离散点，范围从 0 到 1

# 生成四个随机的高斯分布
for _ in range(4):
    # 随机生成中心点 (cx) 和幅值 (A)
    cx = np.random.rand()  # 中心点在 [0, 1] 范围内随机
    A = np.random.rand()  # 幅值在 [0, 1] 范围内随机

    # 生成高斯分布
    sigma = 0.1  # 高斯分布的标准差（控制宽度）
    Z = A * np.exp(-((x - cx) ** 2) / (2 * sigma ** 2))

    # 随机生成颜色
    color = np.random.rand(3)  # 随机生成 RGB 颜色

    # 绘制离散的高斯分布
    ax.stem(x, Z, linefmt=f'C{_}-', markerfmt=f'C{_}o', basefmt=" ")

    # 随机生成中心点 (cx) 和幅值 (A)
    cx1 = np.random.rand()  # 中心点在 [0, 1] 范围内随机
    A1 = np.random.rand()  # 幅值在 [0, 1] 范围内随机

    # 生成高斯分布
    sigma = 0.1  # 高斯分布的标准差（控制宽度）
    Z1 = A1 * np.exp(-((x - cx1) ** 2) / (2 * sigma ** 2))

    # 随机生成颜色
    color = np.random.rand(3)  # 随机生成 RGB 颜色

    # 绘制离散的高斯分布
    ax1.stem(x, Z1, linefmt=f'C{_}-', markerfmt=f'C{_}o', basefmt=" ")

    cx2 = (cx + cx1) / 2
    Z2 = (Z + Z1) / 2

    # 随机生成颜色
    color = np.random.rand(3)  # 随机生成 RGB 颜色

    # 绘制离散的高斯分布
    ax2.stem(x, Z2, linefmt=f'C{_}-', markerfmt=f'C{_}o', basefmt=" ")

# 显示图像
plt.show()
