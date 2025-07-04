import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

x = np.linspace(0, 1, 100)
gammas = [1.0, 1.5, 2.0, 2.5, 3.0]

plt.figure(figsize=(10, 6))

for gamma in gammas:
    y = (1 - x) ** gamma
    line, = plt.plot(x, y, label=f'γ = {gamma}', linewidth=2)
    if gamma in [1.0, 1.5]:
        for x_val in [0.2, 0.8]:
            y_val = (1 - x_val) ** gamma
            # 绘制散点，颜色与曲线一致，带黑色边框
            plt.scatter(x_val, y_val, color=line.get_color(), s=80,
                        edgecolor='black', zorder=5)
            # 调整文本位置避免重叠
            x_offset = 0.03 if x_val == 0.2 else -0.03
            y_offset = 0.03 if y_val > 0.1 else -0.03
            plt.text(x_val + x_offset, y_val + y_offset,
                     f'({x_val}, {y_val:.2f})',
                     fontsize=8, ha='left' if x_val == 0.2 else 'right',
                     va='bottom' if y_val > 0.1 else 'top')

plt.xlabel('IoU', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()