import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAttentionBranch(nn.Module):
    """
    Parallel edge detection branch — reads block1 features, outputs edge logit.
    PARALLEL means:
        - It reads block1 output (x), but x itself is NOT modified.
        - Skip connections passed to the decoder stay clean.
        - Only lateral_edge is changed by this module.
    """

    def __init__(self, in_channels=256, mid_channels=64):
        """
        Args:
            in_channels:  Input feature channels. Must match block1 output.
                          With width_factor=1: block1 output = 64*4 = 256ch.
            mid_channels: Reduced channel count for intermediate layers (64).
        """
        super().__init__()

        # Stage 1 — channel reduction: 256 → 64 via 1×1 conv
        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 2 — spatial feature extraction: 64 → 64 via 3×3 conv
        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 3 — further spatial refinement: 64 → 64 via 3×3 conv
        self.edge_conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Stage 4 — edge logit output: 64 → 1 via 3×3 conv
        self.edge_out = nn.Conv2d(mid_channels, 1, kernel_size=3,
                                   padding=1, bias=True)

    def forward(self, feat, target_size):
        """
        Args:
            feat:        Block1 encoder features [B, 256, Hf, Wf].
                         This tensor is READ ONLY — it is not modified.
            target_size: (H, W) of the original input image.

        Returns:
            lateral_edge: [B, 1, H, W] — full resolution raw edge logit.
        """
        e1 = self.edge_conv1(feat)            # [B, 64,  Hf, Wf]
        e2 = self.edge_conv2(e1)              # [B, 64,  Hf, Wf]
        e3 = self.edge_conv3(e2)              # [B, 64,  Hf, Wf]
        lateral_edge = self.edge_out(e3)      # [B,  1,  Hf, Wf]

        # Upsample to full image resolution.
        lateral_edge = F.interpolate(
            lateral_edge,
            size=target_size,        # (H, W) of the original input
            mode='bilinear',
            align_corners=False
        )
        return lateral_edge
