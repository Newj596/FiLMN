# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from einops import rearrange
import random
import numpy as np
import cv2
from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)
from filters import *

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output

        # logits_ = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # logits = x[i][..., 5:]
                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
                # logits_.append(logits.view(bs, -1, self.no - 5))
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        # return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


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
        # self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        # result = out * self.sa(out)
        return out


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class ECA_Block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_Block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 3
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


# class MoEGate(nn.Module):
#     """动态门控网络，生成专家权重"""
#
#     def __init__(self, in_channels, num_experts, temperature=1.0):
#         super().__init__()
#         self.num_experts = num_experts
#         self.temperature = temperature
#
#         # 门控网络：输入特征 → 专家权重
#         self.gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(in_channels, num_experts)
#         )
#
#     def forward(self, x, use_gumbel=False):
#         # 计算原始权重 [B, num_experts]
#         logits = self.gate(x)
#
#         # Gumbel-Softmax（可选的离散路由）
#         if use_gumbel:
#             weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=1)
#         else:
#             weights = F.softmax(logits / self.temperature, dim=1)
#
#         return weights

# class MoEGate(nn.Module):
#     """动态门控网络，生成专家权重（Top1 MoE 版本）"""
#
#     def __init__(self, in_channels, num_experts, temperature=10.0, training_state=True):
#         super().__init__()
#         self.num_experts = num_experts
#         self.temperature = temperature
#         self.training = training_state
#
#         # 门控网络：输入特征 → 专家权重
#         self.gate = nn.Linear(in_channels, num_experts)
#
#     def forward(self, x, use_gumbel=False):
#         # 计算原始权重 [B, num_experts]
#         x = nn.AdaptiveAvgPool2d(1)(x)
#         x = nn.Flatten()(x)
#         logits = self.gate(x)
#
#         if use_gumbel:
#             # Gumbel-Softmax 生成硬性 Top1 路由（可导）
#             weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=1)
#         else:
#             # 使用 Straight-Through Estimator (STE) 生成 Top1 路由
#             soft = F.softmax(logits / self.temperature, dim=1)
#
#             # 生成硬性 one-hot 权重（前向传播时使用）
#             indices = torch.argmax(soft, dim=1, keepdim=True)
#             hard = torch.zeros_like(soft).scatter_(1, indices, 1.0)
#
#             # 直通估计器：前向用 hard，反向用 soft 的梯度
#             weights = hard - soft.detach() + soft
#
#         return weights

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


class ViTExpert(nn.Module):

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTMoE(nn.Module):
    """ViT-based MoE模块"""

    def __init__(self, dim, num_experts=5, top_k=2, num_heads=8, mlp_dim=1024):
        super().__init__()
        self.experts = nn.ModuleList([
            ViTExpert(dim, num_heads, mlp_dim) for _ in range(num_experts)
        ])
        self.gate = MoEGate(1280, 5, 1.0)
        self.top_k = top_k
        self.dim = dim

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W

        # 特征序列化 (保持原始维度信息)
        x_patches = x.view(B, C, seq_len).permute(0, 2, 1)  # [B, seq_len, C]

        gate_logits = self.gate(x)  # [B, num_experts]
        gate_weights = torch.softmax(gate_logits, dim=-1)

        # 专家选择 (带安全机制)
        top_k_vals, top_k_indices = torch.topk(
            gate_weights,
            k=self.top_k,
            dim=-1,
            sorted=False
        )

        # 权重归一化 (防止除零)
        top_k_weights = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-6)

        # 初始化输出容器
        combined = torch.zeros_like(x_patches)
        expert_counts = torch.zeros(B, device=x.device)

        # 专家计算 (向量化实现)
        for expert_idx, expert in enumerate(self.experts):
            # 创建样本选择掩码
            mask = (top_k_indices == expert_idx).any(dim=1)
            selected_batches = torch.where(mask)[0]

            if selected_batches.numel() == 0:
                continue

            # 获取对应权重
            pos_mask = top_k_indices[mask] == expert_idx
            expert_pos = pos_mask.float().argmax(dim=1)  # 每个样本中该专家的位置
            weights = top_k_weights[mask].gather(1, expert_pos.unsqueeze(1)).squeeze(1)

            # 专家计算
            expert_input = x_patches[selected_batches]
            expert_output = expert(expert_input)  # [num_selected, seq_len, C]

            # 加权输出
            weighted_output = expert_output * weights.view(-1, 1, 1)

            # 使用散射相加实现高效聚合
            combined.index_add_(0, selected_batches, weighted_output)
            expert_counts.index_add_(0, selected_batches, torch.ones_like(selected_batches, dtype=torch.float))

        # 归一化处理 (带安全机制)
        combined = combined / (expert_counts.view(-1, 1, 1) + 1e-6)

        # 恢复空间维度
        output = combined.permute(0, 2, 1).view(B, C, H, W)
        return output


class NonLocalBlock(nn.Module):
    """轻量化非局部模块"""

    def __init__(self, channel):
        super().__init__()
        self.conv_q = nn.Conv2d(channel, channel // 2, 1)
        self.conv_k = nn.Conv2d(channel, channel // 2, 1)
        self.conv_v = nn.Conv2d(channel, channel, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.conv_q(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C/2]
        k = self.conv_k(x).view(B, -1, H * W)  # [B, C/2, HW]
        v = self.conv_v(x).view(B, -1, H * W)  # [B, C, HW]

        att = self.softmax(torch.bmm(q, k))  # [B, HW, HW]
        out = torch.bmm(v, att.permute(0, 2, 1)).view(B, C, H, W)
        return out + x


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def __init__(self):
        super(BaseModel, self).__init__()
        # self.moe = ViTMoE(dim=1280)
        # self.num_experts = 5
        self.gate = MoEGate(1280, self.num_experts, 1)
        # self.triplet01 = TripletAttention()
        # self.triplet02 = TripletAttention()
        # self.triplet03 = TripletAttention()
        # self.triplet04 = TripletAttention()
        # self.triplet05 = TripletAttention()
        # self.triplets = [self.triplet01, self.triplet02, self.triplet03, self.triplet04, self.triplet05]
        # self.eca01 = ECA_Block(512)
        # self.eca02 = ECA_Block(512)
        # self.eca03 = ECA_Block(512)
        # self.eca04 = ECA_Block(512)
        # self.eca05 = ECA_Block(512)
        # self.ecas = [self.eca01, self.eca02, self.eca03, self.eca04, self.eca05]
        # self.mea01 = MEA_Block(512, 8)
        # self.mea02 = MEA_Block(512, 8)
        # self.mea03 = MEA_Block(512, 8)
        # self.mea04 = MEA_Block(512, 8)
        # self.mea05 = MEA_Block(512, 8)
        # self.meas = [self.mea01, self.mea02, self.mea03, self.mea04, self.mea05]
        # self.ca01 = CA_Block(512, 16)
        # self.ca02 = CA_Block(512, 16)
        # self.ca03 = CA_Block(512, 16)
        # self.ca04 = CA_Block(512, 16)
        # self.ca05 = CA_Block(512, 16)
        # self.cas=[self.ca01,self.ca02,self.ca03,self.ca04,self.ca05]
        self.cbam1 = CBAM(1280, 16, 3)
        self.cbam2 = CBAM(1280, 16, 3)
        self.cbam3 = CBAM(1280, 16, 3)
        self.cbam4 = CBAM(1280, 16, 3)
        self.cbam5 = CBAM(1280, 16, 3)
        # self.cbam11 = CBAM(256, 16, 3)
        # self.cbam12 = CBAM(256, 16, 3)
        # self.cbam13 = CBAM(256, 16, 3)
        # self.cbam14 = CBAM(256, 16, 3)
        # self.cbam15 = CBAM(256, 16, 3)
        # self.cbam21 = CBAM(512, 16, 3)
        # self.cbam22 = CBAM(512, 16, 3)
        # self.cbam23 = CBAM(512, 16, 3)
        # self.cbam24 = CBAM(512, 16, 3)
        # self.cbam25 = CBAM(512, 16, 3)
        self.cbams = [self.cbam1, self.cbam2, self.cbam3, self.cbam4, self.cbam5]
        # self.cbams = [[self.cbam01, self.cbam02, self.cbam03, self.cbam04, self.cbam05],
        #               [self.cbam11, self.cbam12, self.cbam13, self.cbam14, self.cbam15],
        #               [self.cbam21, self.cbam22, self.cbam23, self.cbam24, self.cbam25]]

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        mean = torch.mean(x, dim=(0, 1, 2))
        std = torch.std(x, dim=(0, 1, 2))
        threshold = mean + std
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            if m.i == 9:
                batch_size = x.size(0)
                processed = []
                for i in range(batch_size):
                    x_i = x[i]  # (512,8,8)
                    threshold_i = threshold[i].item()
                    index = int(threshold_i / 0.2)
                    index = max(0, min(index, 4))
                    # print(index)
                    cbam = self.cbams[index]
                    x_processed = cbam(x_i.unsqueeze(0))  # (1,512,8,8)
                    processed.append(x_processed)
                x = torch.cat(processed, dim=0)
            # if m.i == 9:
            #     weights = self.gate(x)  # Top1 one-hot 权重
            #
            #     # 计算所有 ECA 专家的输出 [B, num_experts, C, H, W]
            #     expert_outputs = torch.stack([cbam(x) for cbam in self.cbams], dim=1)
            #
            #     # 使用权重融合专家输出
            #     weights = weights.view(x.shape[0], self.num_experts, 1, 1, 1)  # 扩展维度
            #     output = (expert_outputs * weights).sum(dim=1)
            #     x = output
            # if m.i == 9:
            #     x = self.moe(x)
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    def diversity_reward(self, expert_outputs):
        """
        处理五维输入张量的多样性奖励计算
        Args:
            expert_outputs: [batch_size, num_experts, channels, height, width]
        """
        B, E, C, H, W = expert_outputs.shape

        # 特征空间聚合方案
        # 方案1：全局通道压缩（保留空间信息）
        spatial_pool = expert_outputs.mean(dim=2)  # [B, E, H, W]
        flattened = spatial_pool.view(B, E, -1)  # [B, E, H*W]

        # 方案2：空间-通道联合压缩（推荐）
        pooled = expert_outputs.view(B, E, C, -1).mean(dim=-1)  # [B, E, C]

        # 特征归一化
        normalized = F.normalize(pooled, p=2, dim=-1)  # [B, E, C]

        # 相似度矩阵计算
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))  # [B, E, E]

        # 自相似度掩码
        mask = 1 - torch.eye(E, device=expert_outputs.device).unsqueeze(0)  # [1, E, E]
        avg_sim = (similarity * mask).sum() / (E * (E - 1) * B)

        return avg_sim

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 detection model class for object detection tasks, supporting custom configurations and anchors."""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            def _forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))[0]])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5: 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    """YOLOv5 segmentation model for object detection and segmentation tasks with configurable parameters."""

    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 classification model for image classification tasks, initialized with a config file or detection model."""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None


def parse_model(d, ch):
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
