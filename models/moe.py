import torch
from torch import nn
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.threshold = nn.Parameter(torch.tensor(0.5))
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        # out = torch.where(sum_out > self.threshold, sum_out, 0.0)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(x.shape, max_out.shape, x0.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class MoEGate(nn.Module):
    """Dynamic gating network for generating expert weights (Noisy Top1 MoE version)."""

    def __init__(self, in_channels, num_experts, temperature=1.0, training_state=True, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.training = training_state
        self.k = k  # Number of top experts to select

        # Gating network: input features → expert weights
        self.gate = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        # Compute raw logits [B, num_experts]
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = nn.Flatten()(x)
        logits = self.gate(x)

        if self.training:
            # Add noise to the logits
            noise = torch.randn_like(logits) * F.softplus(logits / self.temperature)
            logits = logits + noise

        # Select top-k elements
        topk_values, topk_indices = torch.topk(logits, k=self.k, dim=1)

        # Create a boolean mask for the top-k elements
        mask = torch.zeros_like(logits, dtype=torch.bool)  # 使用布尔类型
        mask.scatter_(1, topk_indices, True)  # 直接标记 True/False

        masked_logits = logits.masked_fill(~mask, -float('inf'))  # 未被选中的位置设为 -inf

        # 应用 Softmax
        weights = F.softmax(masked_logits / self.temperature, dim=1)  # 输出中未被选中的位置权重严格为 0

        return weights


class MoE(nn.Module):
    """YOLOv5 base model."""

    def __init__(self):
        super(MoE, self).__init__()
        self.num_experts = 5
        self.gate = MoEGate(1280, self.num_experts, 1)

        self.cbam1 = CBAM(1280, 16, 3)
        self.cbam2 = CBAM(1280, 16, 3)
        self.cbam3 = CBAM(1280, 16, 3)
        self.cbam4 = CBAM(1280, 16, 3)
        self.cbam5 = CBAM(1280, 16, 3)

        self.cbams = [self.cbam1, self.cbam2, self.cbam3, self.cbam4, self.cbam5]

    def forward(self, x):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        weights = self.gate(x)  # Top1 one-hot 权重

        # 计算所有 ECA 专家的输出 [B, num_experts, C, H, W]
        expert_outputs = torch.stack([cbam(x) for cbam in self.cbams], dim=1)

        # 使用权重融合专家输出
        weights = weights.view(x.shape[0], self.num_experts, 1, 1, 1)  # 扩展维度
        x = (expert_outputs * weights).sum(dim=1)
        return x
