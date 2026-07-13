import torch
import torch.nn as nn
from models.common import Bottleneck

#本文所提出的 GC-DAF 模块在实际代码实现中基于原有 C3GC.py 文件框架进行开发，文
# 件命名及类名称为便于兼容原有 YOLOv5 网络加载机制，仍保留 C3GC 命名。但模块实际结构已进行了全面重构，
# 与原 C3GC 模块在建模思路与功能实现上存在本质区别，故在论文中正式命名为 GC-DAF。
def autopad(k, p=None):  # 自动padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):  # 基础Conv块
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EnhancedCB2d(nn.Module):  # GCNet风格上下文建模模块 DAFBlock；在训练时使用的是EnhancedCB2d名字，实际为DAFBlock新开发模块
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = max(8, inplanes // 4)

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, self.planes, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, inplanes, 1)
        )

        self.local_context = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, groups=inplanes),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 1)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(inplanes * 2, inplanes, kernel_size=1),
            nn.BatchNorm2d(inplanes),
            nn.SiLU()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(inplanes, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        gc = self.global_context(x)
        lc = self.local_context(x)
        fusion = torch.cat([gc.expand_as(x), lc], dim=1)
        fused = self.fuse(fusion)
        spatial_weight = self.spatial_attn(x)
        return (x + fused) * spatial_weight * self.scale + self.shift


class GateMod(nn.Module):  # 残差门控模块
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        return x1 + (x2 - x1) * self.gate(x1 + x2)


class C3GC(nn.Module):  # 主干模块：GCNet思想+注意力+双分支 GC-DAF
    def __init__(self, c1, c2, n=3, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.gc = EnhancedCB2d(c1)

        self.branch1 = nn.Sequential(
            Conv(c1, c_, k=1),
            *[Bottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0) for _ in range(n)]
        )

        self.branch2 = nn.Sequential(  # 增强的局部通路
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_, c_, kernel_size=3, padding=1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

        self.fuse = GateMod(c_)
        self.cv3 = Conv(c_, c2, k=1)

        self.attn = nn.Sequential(  # 保留SE通道注意力
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 // 4, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_gc = self.gc(x)
        y1 = self.branch1(x_gc)
        y2 = self.branch2(x_gc)
        fused = self.fuse(y1, y2)
        out = self.cv3(fused)
        return out * self.attn(out)
