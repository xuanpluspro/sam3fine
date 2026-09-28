import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64, non_linearity=F.relu):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.non_linearity = non_linearity

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim] 或 [batch, input_dim]
        residual = x
        x = self.down_proj(x)
        x = self.non_linearity(x)
        x = self.up_proj(x)
        return x + residual  # 残差连接