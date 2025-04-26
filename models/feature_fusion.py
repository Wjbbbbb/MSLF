import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

    def forward(self, features):
        # 多尺度特征融合，可能包括变形卷积与通道注意力
        fused_features = self.multi_scale_fusion(features)
        return fused_features

    def multi_scale_fusion(self, features):
        # 定义多尺度特征融合
        fused = torch.cat(features, dim=1)  # 简单示例，实际可能更复杂
        return fused
