import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)               # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x, H, W
    
class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_)
            x_ = x_.flatten(2).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class PVTBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads, sr_ratio)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x
    
class PyramidVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1 = PatchEmbed(3, 64, kernel_size=7, stride=4)
        self.stage2 = PatchEmbed(64, 128, kernel_size=3, stride=2)
        self.stage3 = PatchEmbed(128, 256, kernel_size=3, stride=2)
        self.stage4 = PatchEmbed(256, 512, kernel_size=3, stride=2)

        self.blocks1 = nn.ModuleList([PVTBlock(64, 1, sr_ratio=8)])
        self.blocks2 = nn.ModuleList([PVTBlock(128, 2, sr_ratio=4)])
        self.blocks3 = nn.ModuleList([PVTBlock(256, 4, sr_ratio=2)])
        self.blocks4 = nn.ModuleList([PVTBlock(512, 8, sr_ratio=1)])

    def forward(self, x):
        features = []

        x, H, W = self.stage1(x)
        for blk in self.blocks1:
            x = blk(x, H, W)
        features.append(x)

        x = x.transpose(1, 2).reshape(-1, 64, H, W)
        x, H, W = self.stage2(x)
        for blk in self.blocks2:
            x = blk(x, H, W)
        features.append(x)

        x = x.transpose(1, 2).reshape(-1, 128, H, W)
        x, H, W = self.stage3(x)
        for blk in self.blocks3:
            x = blk(x, H, W)
        features.append(x)

        x = x.transpose(1, 2).reshape(-1, 256, H, W)
        x, H, W = self.stage4(x)
        for blk in self.blocks4:
            x = blk(x, H, W)
        features.append(x)

        return features
    
model = PyramidVisionTransformer()
img = torch.randn(1, 3, 224, 224)
outputs = model(img)

for i, feat in enumerate(outputs):
    print(f"Stage {i+1} output shape:", feat.shape)

