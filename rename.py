import os


def rename_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查是否为.jpg文件且包含'_RGB'
        if filename.endswith('.jpg') and '_PreviewData' in filename:
            # 构建旧文件完整路径
            old_path = os.path.join(folder_path, filename)

            # 生成新文件名（替换'_RGB'为空）
            new_filename = filename.replace('_PreviewData', '')
            new_path = os.path.join(folder_path, new_filename)

            try:
                # 执行重命名操作
                os.rename(old_path, new_path)
                print(f"成功重命名: {filename} → {new_filename}")
            except Exception as e:
                print(f"重命名失败: {filename} - 错误信息: {str(e)}")


if __name__ == "__main__":
    # 替换为你的文件夹路径（例如：r'C:\MyImages'）
    target_folder = r'D:\Fusion\FLIR_Aligned-20250305T071216Z-001\FLIR_Aligned\images_thermal_test\data'

    # 验证路径是否存在
    if os.path.exists(target_folder):
        rename_files(target_folder)
        print("处理完成！")
    else:
        print("错误：指定的文件夹路径不存在！")