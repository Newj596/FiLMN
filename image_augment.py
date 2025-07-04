import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class WeatherAugmentation:
    def __init__(self,
                 rain_prob=0.3,
                 snow_prob=0.3,
                 dark_prob=0.3,
                 fog_prob=0.3):
        self.rain_prob = rain_prob
        self.snow_prob = snow_prob
        self.dark_prob = dark_prob
        self.fog_prob = fog_prob

    def apply(self, img):
        """
        输入: uint8格式的BGR图像 (H, W, C)
        输出: uint8格式的增强后BGR图像
        """
        # 转换为float32并归一化
        img = img.astype(np.float32) / 255.0

        # 应用随机效果
        if np.random.rand() < self.rain_prob:
            img = self.add_rain(img)

        if np.random.rand() < self.snow_prob:
            img = self.add_snow(img)

        if np.random.rand() < self.dark_prob:
            img = self.adjust_darkness(img)

        if np.random.rand() < self.fog_prob:
            img = self.add_fog(img)

        # 转换回uint8
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    def add_rain(self, img):
        """
        改进的雨效：
        - 多角度雨丝叠加
        - 动态模糊效果
        - 随机透明度变化
        """
        h, w = img.shape[:2]
        intensity = np.random.uniform(0.1, 0.4)

        # 生成基础雨层
        base = np.zeros((h, w), dtype=np.float32)

        # 生成多方向雨丝
        for _ in range(3):  # 叠加3层不同方向的雨
            angle = np.random.uniform(75, 105)  # 主要垂直方向
            length = np.random.randint(15, 30)

            # 创建运动模糊核
            kernel = self._motion_kernel(angle, length)
            kernel_size = max(kernel.shape)
            padding = kernel_size // 2

            # 生成随机噪声
            noise = np.random.rand(h + padding * 2, w + padding * 2) * 2.5
            noise = cv2.filter2D(noise, -1, kernel)[padding:-padding, padding:-padding]

            # 阈值处理
            rain_layer = np.clip(noise, 0.7, 1.0) - 0.7
            base += rain_layer * np.random.uniform(0.3, 0.7)

        # 标准化并应用
        base = np.clip(base / base.max(), 0, 1)
        rain_layer = base[..., None] * intensity
        return np.clip(img + rain_layer, 0, 1)

    def _motion_kernel(self, angle, length=20):
        """生成运动模糊核"""
        kernel = np.zeros((length, length))
        center = (length - 1) / 2
        radius = length // 2

        # 绘制线段
        angle_rad = np.deg2rad(angle)
        dx = radius * np.cos(angle_rad)
        dy = radius * np.sin(angle_rad)

        cv2.line(kernel,
                 (int(center - dx), int(center - dy)),
                 (int(center + dx), int(center + dy)),
                 1, thickness=1)
        return kernel / kernel.sum()

    def add_snow(self, img):
        """
        改进的雪效：
        - 不同大小的雪花
        - 多层模糊效果
        - 动态飘落方向
        """
        h, w = img.shape[:2]

        # 生成雪花基础层
        layer = np.zeros((h, w), dtype=np.float32)

        # 生成不同尺度的雪花
        for scale in [2.0, 1.0, 0.7, 0.4]:  # 大、中、小三种尺寸
            size = int(min(h, w) * 0.02 * scale)
            density = np.random.uniform(0.001, 0.003)

            # 生成随机点
            points = np.random.rand(h // size, w // size) < density
            points = cv2.resize(points.astype(float), (w, h))

            # 运动模糊
            angle = np.random.uniform(-30, 30)
            kernel = self._motion_kernel(angle, length=int(5 * scale))
            points = cv2.filter2D(points, -1, kernel)

            layer += points * scale  # 尺寸越大亮度越高

        # 模糊处理
        layer = gaussian_filter(layer, sigma=1.5 * scale)
        snow_layer = np.clip(layer, 0, 1)[..., None] * 0.8
        return np.clip(img + snow_layer, 0, 1)

    def adjust_darkness(self, img):
        brightness = np.random.uniform(0.5, 2.0)
        return img ** brightness

    def add_fog(self, img):
        h, w = img.shape[:2]

        # 生成基础参数
        intensity = np.random.uniform(0.3, 0.9)
        fade_start = np.random.uniform(0.1, 0.4)
        fade_range = np.random.uniform(0.3, 0.6)

        # 生成二维坐标网格
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)

        # 改进的Perlin噪声生成
        def fractal_noise(size, octaves=4):
            noise = np.zeros(size)
            for i in range(octaves):
                scale = 2 ** i
                x = np.linspace(0, scale, size[1])
                y = np.linspace(0, scale, size[0])
                X, Y = np.meshgrid(x, y)
                layer = np.sin(2 * np.pi * X + 3 * np.pi * Y) * 0.5 ** i
                noise += layer
            return (noise - noise.min()) / (noise.max() - noise.min())

        # 生成分形噪声（形状为hxw）
        noise = fractal_noise((h, w))

        # 深度渐变（使用二维坐标）
        depth_mask = np.exp(-5 * np.clip(yy - fade_start, 0, 0.2) / fade_range)

        # 合成雾层（形状hxw）
        fog_density = (depth_mask * 0.7 + noise * 0.3) * intensity
        fog_density = gaussian_filter(fog_density, sigma=2)[..., None]  # 变为hxwx1

        # 动态雾色（形状hxwx3）
        base_color = np.random.uniform([0.85, 0.85, 0.85], [0.95, 0.9, 0.95])  # 随机基础色
        height_color = np.array([0.8, 0.85, 0.9]) * (1 - yy[..., None])  # 高度相关颜色
        fog_color = base_color * (1 - depth_mask[..., None]) + height_color * depth_mask[..., None]

        # 物理混合公式
        atmospheric = np.exp(-2 * fog_density)  # 形状hxwx1
        blended = img * atmospheric + fog_color * (1 - atmospheric)

        # 空气光散射（各通道单独处理）
        airlight = np.stack([gaussian_filter(img[..., c], sigma=5) for c in range(3)], axis=-1) * 0.3
        final = blended * 0.7 + airlight * 0.3

        return np.clip(final, 0, 1)


# 图像处理与可视化
def visualize_effects(image_path):
    # 读取图像
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # 初始化增强器
    augmenter = WeatherAugmentation(
        rain_prob=0.0,
        snow_prob=0.0,
        dark_prob=0.0,
        fog_prob=1.0
    )

    # 创建画布
    plt.figure(figsize=(20, 10))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis('off')

    # 雨效
    rain_img = augmenter.apply(orig_img.copy())
    plt.subplot(2, 3, 2)
    plt.imshow(rain_img)
    plt.title("Rain Effect")
    plt.axis('off')

    # 雪效
    snow_img = augmenter.apply(orig_img.copy())
    plt.subplot(2, 3, 3)
    plt.imshow(snow_img)
    plt.title("Snow Effect")
    plt.axis('off')

    # 暗光
    dark_img = augmenter.apply(orig_img.copy())
    plt.subplot(2, 3, 4)
    plt.imshow(dark_img)
    plt.title("Darkness Effect")
    plt.axis('off')

    # 雾效
    fog_img = augmenter.apply(orig_img.copy())
    plt.subplot(2, 3, 5)
    plt.imshow(fog_img)
    plt.title("Fog Effect")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    image_path = r"D:\Writing\datasets\Fog\images\train_norm\000132.jpg"  # 替换为你的图片路径
    visualize_effects(image_path)
