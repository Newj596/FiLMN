import cv2
from models.gradcam import YOLOV5GradCAM
from models.yolov5_object_detector import YOLOV5TorchObjectDetector

# 初始化YOLOv5模型
model = YOLOV5TorchObjectDetector('yolov5s.pt', 'cuda', (640, 640))

# 加载图像并进行预处理
img = cv2.imread('data/images/bus.jpg')
torch_img = model.preprocessing(img[..., ::-1]) # BGR to RGB

# 使用Grad-CAM
saliency_method = YOLOV5GradCAM(model=model, layer_name='model_17_cv3_act', img_size=(640, 640))
masks, logits, _ = saliency_method(torch_img)

# 叠加热力图和原始图像
res_img = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
res_img = res_img[..., ::-1] # RGB to BGR
heatmap = cv2.applyColorMap(masks[0].cpu().numpy(), cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + res_img

# 显示结果
cv2.imshow('YOLOv5 Heatmap Visualization', superimposed_img.astype('uint8'))
cv2.waitKey(0)