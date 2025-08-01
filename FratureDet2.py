import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.GroupNorm(32, in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            # Softmax over spatial dimensions: height * width
            # We'll handle reshaping in forward
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        # Apply convolution layers
        attention = self.conv(x)  # (batch, channels, height, width)
        # Reshape to (batch, channels, height*width)
        attention = attention.view(batch, channels, -1)
        # Apply softmax over spatial dimensions
        attention = F.softmax(attention, dim=-1)
        # Reshape back to (batch, channels, height, width)
        attention = attention.view(batch, channels, height, width)
        return x * attention

class EnhancedFPNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.depth_conv = nn.Conv2d(out_channels, out_channels, 
                                   kernel_size=3, padding=1, groups=out_channels)
        self.point_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.spatial_attention = SpatialAttention(out_channels)
        self.gn = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.spatial_attention(x)
        x = self.gn(x)
        return self.relu(x)

class EnhancedFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.fpn_layers = nn.ModuleList([
            EnhancedFPNLayer(out_channels, out_channels) for _ in in_channels_list
        ])

    def forward(self, inputs):
        """
        inputs: list of feature maps [C5, C4, C3, C2] from backbone
        """
        lateral_features = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, inputs)]
        
        # Build top-down path
        for i in range(1, len(lateral_features)):
            # Get the spatial size of the previous feature map
            target_size = lateral_features[i-1].shape[2:]
            # Upsample the current feature map to the target size
            upsampled = F.interpolate(lateral_features[i], size=target_size, mode='nearest')
            # Add the upsampled feature map to the previous one
            lateral_features[i-1] += upsampled
        
        # Apply FPN layers
        fpn_features = [fpn_layer(f) for fpn_layer, f in zip(self.fpn_layers, lateral_features)]
        return fpn_features  # [P5, P4, P3, P2]

class SpatialRegressionHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 5)  # <--- 多加一層
        )

    def forward(self, x):
        return self.fc(x)

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_dim1=512, hidden_dim2=256, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x, dropout_p1=0.3, dropout_p2=0.2):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = F.dropout(x, p=dropout_p1, training=self.training)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = F.dropout(x, p=dropout_p2, training=self.training)
        x = self.fc3(x)
        return x

class ResNet152RotatedBBoxNet_5d(nn.Module):
    def __init__(self, backbone='resnet152', fpn_channels=256, reg_dropout_p=0.3, cls_dropout_p1=0.3, cls_dropout_p2=0.2):
        super().__init__()
        self.cls_dropout_p1 = cls_dropout_p1
        self.cls_dropout_p2 = cls_dropout_p2
        self.reg_dropout_p = reg_dropout_p

        # Load backbone
        backbone_model = getattr(models, backbone)(pretrained=True)
        self.conv1 = backbone_model.conv1
        self.bn1 = backbone_model.bn1
        self.relu = backbone_model.relu
        self.maxpool = backbone_model.maxpool

        self.layer1 = backbone_model.layer1  # C2: 256 channels
        self.layer2 = backbone_model.layer2  # C3: 512 channels
        self.layer3 = backbone_model.layer3  # C4: 1024 channels
        self.layer4 = backbone_model.layer4  # C5: 2048 channels

        # Enhanced FPN
        in_channels_list = [2048, 1024, 512, 256]
        self.fpn = EnhancedFPN(in_channels_list, fpn_channels)
        
        total_fpn_features = fpn_channels * 4

        # 使用 ClassificationHead 類別
        self.cls_head = ClassificationHead(total_fpn_features, hidden_dim1=512, hidden_dim2=256, num_classes=2)
        
        self.reg_head = SpatialRegressionHead(total_fpn_features, hidden_dim=256, dropout_p=reg_dropout_p)

    def _global_pool(self, x):
        return torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)

    def forward(self, x, cls_dropout_p1=None, cls_dropout_p2=None, reg_dropout_p=None):
        """
        前向傳播，支持動態dropout率
        Args:
            x: 輸入張量
            cls_dropout_p1: 分類分支第一層的dropout率，如果為None則使用初始化時的值
            cls_dropout_p2: 分類分支第二層的dropout率，如果為None則使用初始化時的值
            reg_dropout_p: 回歸分支的dropout率，如果為None則使用初始化時的值
        """
        # 使用提供的dropout率或默認值
        cls_drop1 = self.cls_dropout_p1 if cls_dropout_p1 is None else cls_dropout_p1
        cls_drop2 = self.cls_dropout_p2 if cls_dropout_p2 is None else cls_dropout_p2
        reg_drop = self.reg_dropout_p if reg_dropout_p is None else reg_dropout_p

        # Backbone特徵提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 獲取各個階段的特徵圖
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN特徵提取
        fpn_features = self.fpn([c5, c4, c3, c2])  # [P5, P4, P3, P2]

        # 全局池化
        pooled_features = [self._global_pool(p) for p in fpn_features]
        
        # 連接所有特徵
        feat = torch.cat(pooled_features, dim=1)
        
        # 分類分支前向傳播
        logits_cls = self.cls_head(feat, dropout_p1=cls_drop1, dropout_p2=cls_drop2)
        
        # 回歸分支前向傳播
        bbox_5d = self.reg_head(feat)
        
        return logits_cls, bbox_5d





