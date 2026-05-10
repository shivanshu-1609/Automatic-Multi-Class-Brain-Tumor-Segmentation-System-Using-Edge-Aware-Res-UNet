from collections import OrderedDict
import torch
import torch.nn as nn
from models.utils import StdConv2d, conv1x1, conv3x3, np2th
from models.edge_branch import EdgeAttentionBranch

class PreActBottleneck(nn.Module):
    """
    Pre-activation bottleneck block (ResNet v2 style).
    Order of operations: Norm → Activation → Conv (pre-activation).
    Structure: cin → cmid (1×1) → cmid (3×3) → cout (1×1) + residual
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        # Pre-normalization and 1×1 channel reduction
        self.gn1  = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)

        # Spatial convolution with optional strided downsampling
        self.gn2  = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)

        # 1×1 channel expansion back to cout
        self.gn3  = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu  = nn.ReLU(inplace=True)

        # Projection shortcut when input/output shapes differ
        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj    = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.gn_proj(self.downsample(x))

        # Pre-activation path: Norm → ReLU → Conv
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)


class ResNetV2(nn.Module):
    """
    Pre-activation ResNet v2 encoder.
    Produces:
        - bottleneck : deepest features [B, 1024, H/16, W/16]
        - features   : skip connections in decoder order (deepest first)
                       [block2(512), block1(256), root(64)]
        - lateral_edge: full-resolution 1-ch edge logit [B, 1, H, W]
    The EdgeAttentionBranch runs IN PARALLEL on block1 output.
    """

    def __init__(self, block_units, width_factor, in_channels=4):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        # Stem
        self.root = nn.Sequential(
            OrderedDict([
                ("conv", StdConv2d(in_channels, width, kernel_size=7,
                                   stride=2, bias=False, padding=3)),
                ("gn",   nn.GroupNorm(32, width, eps=1e-6)),
                ("relu", nn.ReLU(inplace=True)),
            ])
        )

        # Encoder body
        self.body = nn.Sequential(
            OrderedDict([
                ("block1", self._make_block(width,     width * 4,  width,     block_units[0])),
                ("block2", self._make_block(width * 4, width * 8,  width * 2, block_units[1], stride=2)),
                ("block3", self._make_block(width * 8, width * 16, width * 4, block_units[2], stride=2)),
            ])
        )

        # Edge Attention Branch
        self.attention = EdgeAttentionBranch(
            in_channels=width * 4, 
            mid_channels=64
        )

    def _make_block(self, cin, cout, cmid, num_units, stride=1):
        layers = [PreActBottleneck(cin, cout, cmid, stride)]
        for _ in range(1, num_units):
            layers.append(PreActBottleneck(cout, cout, cmid))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        input_size = (x.shape[2], x.shape[3])

        # Stem: stride=2
        x = self.root(x)
        features.append(x)  

        # Pool: stride=2
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)

        # Block 1
        x = self.body[0](x)
        features.append(x)

        # Edge branch runs in PARALLEL on block1 output
        lateral_edge = self.attention(x, target_size=input_size)

        # Block 2
        x = self.body[1](x)
        features.append(x)

        # Block 3 (bottleneck)
        x = self.body[2](x)

        return x, features[::-1], lateral_edge
