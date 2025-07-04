import cv2
import os

# 设置视频路径和输出文件夹
video_path = r'D:\Writing\yolov5\runs\detect\exp3\傍晚雾.mp4'  # 替换为你的视频路径
output_dir = r'D:\Writing\yolov5\runs\detect\exp3\frames'   # 输出文件夹名称

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

frame_count = 0    # 总帧数计数器
save_counter = 0   # 保存图片计数器

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频结束或读取失败时退出循环

    # 每隔30帧保存一次
    if frame_count % 30 == 0:
        # 生成文件名（格式：frame_0000.jpg）
        filename = os.path.join(output_dir, f'frame_{save_counter:04d}.jpg')
        cv2.imwrite(filename, frame)
        print(f'已保存：{filename}')
        save_counter += 1

    frame_count += 1

# 释放资源
cap.release()
print(f'处理完成，共保存 {save_counter} 张图片')