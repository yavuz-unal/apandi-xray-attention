"""
Model architectures and attention mechanisms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# Attention Modules
# ============================================================

class SE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * sa


class CA(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, in_channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class ECA(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(in_channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)


class PCCA(nn.Module):
    """Parallel Coordinate-Channel Attention: CA and ECA branches combined with learnable gate."""
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.ca = CA(in_channels, reduction)
        self.eca = ECA(in_channels)
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)

    def forward(self, x):
        ca_out = self.ca(x)
        eca_out = self.eca(x)
        alpha = torch.sigmoid(self.alpha)
        out = alpha * ca_out + (1 - alpha) * eca_out
        return out + x


# ============================================================
# Backbones
# ============================================================

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, attention=None):
        super().__init__()
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.features = backbone.features
        in_features = backbone.classifier[1].in_features
        self.attn = attention(in_features) if attention else nn.Identity()
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DenseNet121(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.densenet121(weights='IMAGENET1K_V1')
        self.features = backbone.features
        in_features = 1024
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        in_features = 2048
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SwinTiny(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.swin_t(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.norm = backbone.norm
        self.permute = backbone.permute
        in_features = 768
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        self.features = backbone.features
        in_features = 768
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# Factory
# ============================================================

def build_model(name, num_classes=2):
    if name == 'effnet_base':
        return EfficientNetB0(num_classes=num_classes, attention=None)
    elif name == 'effnet_se':
        return EfficientNetB0(num_classes=num_classes, attention=SE)
    elif name == 'effnet_cbam':
        return EfficientNetB0(num_classes=num_classes, attention=CBAM)
    elif name == 'effnet_ca':
        return EfficientNetB0(num_classes=num_classes, attention=CA)
    elif name == 'effnet_pcca':
        return EfficientNetB0(num_classes=num_classes, attention=PCCA)
    elif name == 'densenet_base':
        return DenseNet121(num_classes=num_classes)
    elif name == 'resnet_base':
        return ResNet50(num_classes=num_classes)
    elif name == 'swin_base':
        return SwinTiny(num_classes=num_classes)
    elif name == 'convnext_base':
        return ConvNeXtTiny(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model: {name}')
