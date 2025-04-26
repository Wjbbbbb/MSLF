import torch
import torch.nn as nn


class AttentionGuidedFusion(nn.Module):
    def __init__(self):
        super(AttentionGuidedFusion, self).__init__()
        # 定义注意力模块
        self.attention_layer = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, content, style):
        # 计算注意力，指导融合
        attention_map = torch.sigmoid(self.attention_layer(content))
        fused_output = content * attention_map + style * (1 - attention_map)
        return fused_output
