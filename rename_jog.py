import os
import argparse


def rename_image_extensions(folder_path):
    """
    将文件夹内所有图像文件后缀统一改为.jpg格式
    :param folder_path: 目标文件夹路径
    """
    # 支持的图像格式列表（可扩展）
    image_extensions = ['.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.jpg']

    for filename in os.listdir(folder_path):
        # 获取文件完整路径
        filepath = os.path.join(folder_path, filename)

        # 跳过子目录和非文件项
        if not os.path.isfile(filepath):
            continue

        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        # 跳过非图像文件
        if ext_lower not in image_extensions:
            continue

        # 构造新文件名
        new_filename = name + '.jpg'
        new_filepath = os.path.join(folder_path, new_filename)

        # 跳过已经是.jpg格式的文件
        if new_filename == filename:
            continue

        # 处理文件名冲突
        counter = 1
        while os.path.exists(new_filepath):
            new_filename = f"{name}_{counter}.jpg"
            new_filepath = os.path.join(folder_path, new_filename)
            counter += 1

        # 执行重命名
        try:
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
        except Exception as e:
            print(f"Error renaming {filename}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量修改图像文件后缀为.jpg')
    parser.add_argument('folder', type=str, help='目标文件夹路径')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"错误: 路径 {args.folder} 不是有效文件夹")
    else:
        rename_image_extensions(args.folder)
        print("\n操作完成！建议使用图像查看器验证文件完整性")