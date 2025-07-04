import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_images_grid(images, titles, filename=None):
    """
    将多张图片绘制在一个网格图中
    :param images: 图片列表 (NumPy 数组格式)
    :param titles: 每个子图的标题列表 (可选)
    :param filename: 文件名 (用于大图标题)
    """
    n = len(images)
    if n == 0:
        print("没有可显示的图像")
        return

    # 动态计算行列数 (例如：4张图 → 2x2)
    rows = int(n ** 0.5)
    cols = int(np.ceil(n / rows))

    # 创建子图
    fig, axes = plt.subplots(1, 5, figsize=(12, 5))
    if rows == 1 or cols == 1:
        axes = axes.reshape(-1)  # 确保 axes 是1维数组

    # 遍历每个子图绘制图像
    for idx, ax in enumerate(axes.flatten()):
        if idx < n:
            img = images[idx]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB 转换
            ax.imshow(img_rgb)
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=12, y=-0.15)
            ax.axis('off')
        else:
            ax.axis('off')  # 隐藏多余的子图

    # 设置大标题并调整布局
    # plt.suptitle(f"File: {filename}" if filename else "Image Comparison")
    plt.tight_layout()
    plt.show()


# 配置参数
folder_paths = [  # 替换为你的文件夹路径列表
    r"C:\Users\20466\Desktop\FL-MoE\figures\Fog\Yolo",
    r"C:\Users\20466\Desktop\FL-MoE\figures\Fog\Dehazeformer",
    r"C:\Users\20466\Desktop\FL-MoE\figures\Fog\DSANet",
    r"C:\Users\20466\Desktop\FL-MoE\figures\Fog\FL-MoE",
    r"C:\Users\20466\Desktop\FL-MoE\figures\Fog\Groundtruth"
]
image_extensions = ('.png', '.jpg', '.jpeg', '.gif')  # 支持的图片格式

# 获取所有共同文件名
common_files = None
for folder in folder_paths:
    # 获取当前文件夹所有图片文件
    files = {f for f in os.listdir(folder)
             if f.lower().endswith(image_extensions)}

    if common_files is None:
        common_files = files
    else:
        common_files &= files

# 检查是否有共同文件
if not common_files:
    raise ValueError("没有找到共同图片文件")

sorted_files = sorted(common_files)

# 对每个文件进行操作
for filename in sorted_files[::5]:
    images = []
    for folder in folder_paths:
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise Exception(f"无法加载图像: {img_path}")
            images.append(img)
        except Exception as e:
            print(f"加载失败：{img_path} - {str(e)}")
            continue
    # 显示图片
    # for idx, img in enumerate(images):
    #     cv2.imshow(f'{os.path.basename(folder_paths[idx])}\n{filename}', img)
    # 等待按键以显示下一张图片
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    titles = ["Baseline", "Dehazeformer", "DSANet", "FL-MoE", "GroundTruth"]
    plot_images_grid(images, titles)
