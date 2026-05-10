import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dReLU(nn.Sequential):
    """Standard Conv → BN → ReLU block used in the UNet decoder."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding,
                         bias=not use_batchnorm)
        bn   = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    """
    Single UNet decoder stage: upsample → (optional skip cat) → 2× Conv-BN-ReLU.
    The optional `edge_att` argument applies spatial attention AFTER
    concatenating the skip connection.
    """
    def __init__(self, in_channels, out_channels, skip_channels=0,
                 use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels,
                                kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels,
                                kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None, edge_att=None):
        x = self.up(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                   mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        # EDGE ATTENTION MECHANISM
        if edge_att is not None:
            if edge_att.shape[2:] != x.shape[2:]:
                edge_att = F.interpolate(edge_att, size=x.shape[2:],
                                          mode="bilinear", align_corners=False)
            x = x * edge_att + x

        return x

class SegmentationHead(nn.Sequential):
    """
    Final 1×1 convolutional layer mapping decoder channels → class logits.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d   = nn.Conv2d(in_channels, out_channels,
                             kernel_size=kernel_size,
                             padding=kernel_size // 2)
        upsample = (nn.UpsamplingBilinear2d(scale_factor=upsampling)
                    if upsampling > 1 else nn.Identity())
        super().__init__(conv2d, upsample)


class DecoderCup(nn.Module):
    """
    4-stage UNet decoder with skip connections AND edge attention modulation.
    """
    def __init__(self):
        super().__init__()

        head_channels = 512

        self.conv_more = Conv2dReLU(1024, head_channels, kernel_size=3,
                                    padding=1)

        decoder_channels = (512, 256, 128, 64)
        in_channels      = [head_channels] + list(decoder_channels[:-1])
        skip_channels = [512, 256, 64, 0]

        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in
            zip(in_channels, decoder_channels, skip_channels)
        ])

    def forward(self, hidden_states, features=None, lateral_edge=None):
        x = self.conv_more(hidden_states)

        edge_att = torch.sigmoid(lateral_edge) if lateral_edge is not None else None

        for i, decoder_block in enumerate(self.blocks):
            skip = (features[i]
                    if (features is not None and i < len(features))
                    else None)

            x = decoder_block(x, skip=skip, edge_att=edge_att)

        return x
