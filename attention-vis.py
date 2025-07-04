import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox  # 关键修改点


class HeatmapGenerator:
    def __init__(self, model_weights, device='cuda:0'):
        self.device = torch.device(device)
        self.model = DetectMultiBackend(model_weights, device=self.device, fp16=False)
        self.model.eval()

        self.gradients = None
        self.activations = None

        # 获取目标层（根据模型结构调整）
        target_layer = self.model.model.model[-2]  # 示例：选择倒数第二层
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, img_path, conf_thres=0.25):
        # 图像预处理（修复核心）
        img_orig = cv2.imread(img_path)
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        # 使用 letterbox 进行标准化预处理
        img_processed = letterbox(img, new_shape=self.model.model.stride.max())[0]  # 自动填充
        img_processed = img_processed.transpose(2, 0, 1)  # HWC -> CHW
        img_tensor = torch.from_numpy(img_processed).to(self.device)
        img_tensor = img_tensor.float() / 255.0  # 归一化
        img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度

        # 前向传播
        pred = self.model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=conf_thres)[0]

        # 反向传播（示例：选择第一个检测目标的置信度）
        self.model.zero_grad()
        if pred is not None and len(pred) > 0:
            target_score = pred[0, 4]  # 取第一个目标的置信度
            target_score.backward(retain_graph=True)
        else:
            raise ValueError("No detections found")

        # 计算 Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        weighted_activations = pooled_gradients[:, None, None] * self.activations[0]
        heatmap = torch.mean(weighted_activations, dim=0).cpu().detach().numpy()

        # 后处理热力图
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        return heatmap, img_orig, pred


# 使用示例
generator = HeatmapGenerator('yolov5s.pt')
heatmap, img, detections = generator.generate_heatmap(r"D:\Writing\datasets\Fog\images\test_rtts\NY_Google_111.png")

# 可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('Grad-CAM Heatmap')
plt.axis('off')
plt.show()