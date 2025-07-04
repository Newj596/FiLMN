import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# TRO期刊格式设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'legend.frameon': False
})

# 数据准备
RTTS = [76.5, 75.8, 75.0, 73.9, 72.1, 70.1, 66.8, 61.4, 52.8]
ExDark = [64.5, 65.2, 65.0, 64.5, 63.6, 62.0, 59.3, 54.7, 41.6]

# 转换为小数
RTTS = [i * 0.01 for i in RTTS]
ExDark = [i * 0.01 for i in ExDark]
x = np.linspace(0.1, 0.9, 9)  # 修正为0.1-0.9的均匀分布

# 创建图表
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制折线
ax.plot(x, RTTS, color='#E41A1C', marker='o', markersize=8, label='RTTS')
ax.plot(x, ExDark, color='#377EB8', marker='s', markersize=8, linestyle='--', label='ExDark')

# 坐标轴设置
ax.set_xticks(x)  # 设置精确刻度位置
ax.set_xticklabels([f"{i:.1f}" for i in x])  # 强制显示一位小数
ax.set_xlim(0.05, 0.95)  # 留出边界空白
ax.set_ylim(0.3, 0.8)
ax.set_xlabel('Confidence', fontweight='bold')
ax.set_ylabel('mAP50', fontweight='bold')

# 设置百分比格式
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))  # 显示整数百分比

# 网格和边框设置
ax.grid(True, linestyle='--', alpha=0.6)
for spine in ['top', 'right']:  # 恢复边框设置
    ax.spines[spine].set_visible(False)

# 图例
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

# 保存和显示
plt.tight_layout()
plt.savefig('TRO_style_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()