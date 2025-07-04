import os


def modify_labels(input_dir, output_dir=None):
    """
    修改YOLO格式的标签文件中的类别号，将每个类别号减1。

    :param input_dir: 包含YOLO格式标签文件的文件夹路径
    :param output_dir: 保存修改后的标签文件的文件夹路径，默认为None，表示覆盖原文件
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:  # 确保每行至少有5个部分（类别号 + 4个边界框坐标）
                    continue
                class_id = int(parts[0])
                new_class_id = class_id - 1
                parts[0] = str(new_class_id)
                modified_line = ' '.join(parts) + '\n'
                modified_lines.append(modified_line)

            if output_dir:
                output_file_path = os.path.join(output_dir, filename)
            else:
                output_file_path = file_path

            with open(output_file_path, 'w') as file:
                file.writelines(modified_lines)


# 使用示例
input_directory = r"D:\Writing\datasets\DAWN\labels\test_Rain"  # 替换为你的YOLO标签文件夹路径
output_directory = None  # 替换为你希望保存修改后标签文件的文件夹路径，如果为None则覆盖原文件

modify_labels(input_directory, output_directory)