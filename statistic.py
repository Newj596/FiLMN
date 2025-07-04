# from collections import Counter
#
# # 读取文件并统计数字出现次数
# with open('moe_rtts.txt', 'r') as file:
#     numbers = [line.strip() for line in file if line.strip()]
#
# count = Counter(numbers)
#
# # 按数值大小排序
# sorted_counts = sorted(count.items(), key=lambda item: float(item[0]))
#
# # 输出结果
# for number, frequency in sorted_counts:
#     print(f"{number}: {frequency}")

import matplotlib.pyplot as plt
import numpy as np

# 原始数据
dict_rtts = {0: 3229, 1: 46, 2: 285, 3: 357, 4: 405}
dict_norm = {0: 917, 1: 353, 2: 502, 3: 171, 4: 793}
dict_fog = {0: 1256, 1: 175, 2: 199, 3: 291, 4: 814}

# 提取类别和数值
categories = list(dict_rtts.keys())
rtts_values = list(dict_rtts.values())
norm_values = list(dict_norm.values())
fog_values = list(dict_fog.values())

# 柱状图参数设置
bar_width = 0.25  # 柱子宽度
x = np.arange(len(categories))  # x轴基础位置

# 绘制柱状图
plt.figure(figsize=(12, 6))

# 分别绘制三组柱子
rects1 = plt.bar(x - bar_width, rtts_values, bar_width, label='RTTS', color='#1f77b4', edgecolor='black')
rects2 = plt.bar(x, norm_values, bar_width, label='VOC_Norm1', color='#ff7f0e', edgecolor='black')
rects3 = plt.bar(x + bar_width, fog_values, bar_width, label='VOC_Fog', color='#2ca02c', edgecolor='black')

# 添加标签和标题
plt.xlabel('Category', fontsize=15, fontweight='bold')
plt.ylabel('Count', fontsize=15, fontweight='bold')
# plt.title('Category Distribution Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, categories)
plt.legend(fontsize=15)

# 自动添加柱子顶部的数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=15)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 调整布局
plt.tight_layout()
plt.show()
