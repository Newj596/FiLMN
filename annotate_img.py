import cv2
import os
import numpy as np

# hexs = (
#     # "FF3838",
#     # "FF9D97",
#     # "FF701F",
#     # "FFB21D",
#     "CFD231",
#     # "48F90A",
#     # "92CC17",
#     "3DDB86",
#     # "1A9334",
#     # "00D4BB",
#     # "2C99A8",
#     "00C2FF",
#     # "344593",
#     # "6473FF",
#     # "0018EC",
#     "8438FF",
#     # "520085",
#     # "CB38FF",
#     # "FF95C8",
#     "FF37C7",
# )

hexs = [
    "CF31D2", "3D86DB", "00FFC2", "84FF38", "FFC737",
    "CF69D2", "3DFFDB", "0035C2", "84FF60", "FFC780",
    "CF0AD2", "FFDB86"
]

# classes = ["person", "car", "bus", "bicycle", "motorcycle"]
classes = ["bicycle",
           "boat",
           "bottle",
           "bus",
           "car",
           "cat",
           "chair",
           "dog",
           "motorbike",
           "person"]


def hex2rgb(h):
    """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
    return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (2, 4, 0))


def read_txt_gt(txt_list):
    with open(txt_list, "r") as f:
        labels_list = []
        labels = f.readlines()
        labels = [_[:-1] for _ in labels]
        labels = [_.split(" ") for _ in labels]
        for label in labels:
            label = [float(item) for item in label]
            labels_list.append(label)
    f.close()
    return labels_list


def read_txt_teacher(txt_list):
    contents_list = []
    for i, txt_path in enumerate(txt_list):
        with open(txt_path, "r") as f:
            contents = f.readlines()
            for cont in contents:
                cont = str(i) + " " + cont
                cont = cont[:-1]  # [1+4+n+1] 1: image index 4: box n: class logits 1:cls 1: path w/o .txt
                cont = cont.split(" ")
                cont = [float(con) for con in cont]
                # cont[1:5] = convert_xyxy_to_xywh(cont[1:5])
                contents_list.append(cont)
        f.close()
    return contents_list


img_dir = r"D:\Writing\datasets\Light\images\test_exdark"
label_dir_gt = img_dir.replace("images", "labels")
# label_dir_teacher = img_dir.replace("images", "labels_teacher")

image_gt_dir = r"D:\Writing\vis_results\gt1"
# image_dino_dir = r"D:\Writing\vis_results\dino"

img_list = os.listdir(img_dir)
colors = [hex2rgb(f"#{c}") for c in hexs]
print(colors)

for i, img_name in enumerate(img_list):
    print(i)
    img_path = os.path.join(img_dir, img_name)
    img_name1 = img_name.replace(".jpg", ".txt")
    img_name1 = img_name1.replace(".png", ".txt")
    if img_name1 not in os.listdir(label_dir_gt):
        continue
    label_path_gt = os.path.join(label_dir_gt, img_name1)
    # label_path_teacher = os.path.join(label_dir_teacher, img_name.replace(".jpg", ".txt"))
    # label_path_teacher = os.path.join(label_dir_teacher, img_name.replace(".png", ".txt"))
    labels_gt = read_txt_gt(label_path_gt)
    # labels_teacher = read_txt_teacher([label_path_teacher])

    img_ = cv2.imread(img_path)
    img = img_.copy()
    img0 = img.copy()
    img1 = img.copy()
    h0, w0, c = img0.shape
    # labels_gt0=[[0,0.5,0.4,0.3,0.32],[1,0.62,0.55,0.4,0.25],[1,0.7,0.5,0.52,0.4],[3,0.25,0.22,0.32,0.3],[0,0.12,0.73,0.2,0.26],[0,0.5,0.5,0.33,0.42]]
    # for label in labels_gt0:
    #     class_id = int(label[0])  # 标签类别
    #     x, y, w, h = label[1:5]  # 提取坐标和宽高
    #
    #     # 计算检测框的左上角和右下角坐标
    #     x1, y1 = x - w / 2, y - h / 2  # 左上角
    #     x2, y2 = x + w / 2, y + h / 2  # 右下角
    #     x1 = int(x1 * w0)
    #     y1 = int(y1 * h0)
    #     x2 = int(x2 * w0)
    #     y2 = int(y2 * h0)
    #     # 在图像上绘制矩形框
    #     color = (75, 180, 255)  # 绿色框 (BGR 格式)
    #     thickness = 2  # 线宽
    #     cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    #
    #     # 可选：在框上方显示标签类别
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 0.5
    #     text_color = (0, 255, 0)  # 红色文本
    #     cv2.putText(img, f"{class_id}", (x1, y1 - 5), font, font_scale, text_color, thickness)

    for label in labels_gt:
        class_id = int(label[0])  # 标签类别
        x, y, w, h = label[1:5]  # 提取坐标和宽高

        # 计算检测框的左上角和右下角坐标
        x1, y1 = int((x - w / 2) * w0), int((y - h / 2) * h0)  # 左上角
        x2, y2 = int((x + w / 2) * w0), int((y + h / 2) * h0)  # 右下角

        # 在图像上绘制矩形框
        color = colors[class_id]  # 使用预定义的颜色列表获取颜色
        thickness = 3  # 线宽
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, thickness)

        # 绘制文本标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_thickness = 2
        text_color = (255, 255, 255)  # 白色文本

        # 准备显示的文本
        label_text = classes[class_id]

        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

        # 调整文本起点，确保不超出检测框左边界
        text_x = max(x1, x1 + 5)  # 默认情况下，在检测框内稍微向右移动一点，避免紧贴边框
        text_y = y1 - 3  # 文本位于框上方，且保持一定间隔

        # 如果计算出的文本起点仍可能导致文本超出左边界，则直接使用检测框的左上角x坐标
        if text_x < x1 or text_x < x1 + text_width:
            text_x = x1

        # 在检测框上方绘制背景矩形作为文本背景
        background_padding = 2  # 文本与背景边缘的距离
        background_x1 = text_x
        background_y1 = text_y - text_height - background_padding
        background_x2 = text_x + text_width + background_padding
        background_y2 = text_y + background_padding

        # 绘制背景矩形
        cv2.rectangle(img0, (background_x1, background_y1), (background_x2, background_y2), color, -1)

        # 放置文本
        cv2.putText(img0, label_text, (text_x, text_y), font, font_scale, text_color, text_thickness,
                    lineType=cv2.LINE_AA)
        # cv2.imwrite(os.path.join(image_gt_dir, img_name), img0)

    cv2.imshow("test", img0)
    cv2.waitKey(0)
