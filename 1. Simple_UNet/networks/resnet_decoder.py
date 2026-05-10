from collections import OrderedDict # keeps layer order fixed
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F # functional ops (conv2d, interpolate, etc.)


def np2th(weights, conv=False):
    # Convert numpy weights to torch format (HWIO → OIHW for conv layers)
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    # Standardized convolution (weight normalization before convolution)
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)

        return F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    # 3x3 convolution with padding
    return StdConv2d(
        cin,
        cout,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        groups=groups,
    )


def conv1x1(cin, cout, stride=1, bias=False):
    # 1x1 convolution (channel projection)
    return StdConv2d(
        cin,
        cout,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias,
    )


class PreActBottleneck(nn.Module):
    # Pre-activation ResNet bottleneck block (ResNet v2 style)
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()

        cout = cout or cin
        cmid = cmid or cout // 4

        # Normalization + convolution layers
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)

        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)

        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)

        self.relu = nn.ReLU(inplace=True)

        # Downsampling path for residual connection
     
        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x

        # Match dimensions if needed
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Main path
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        # Residual addition
        y = self.relu(residual + y)

        return y

    def load_from(self, weights, n_block, n_unit):
        # Load pretrained weights (if available)
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)


class ResNetV2(nn.Module):
    # ResNet encoder (pre-activation version)
    def __init__(self, block_units, width_factor, in_channels=4):
        super().__init__()

        width = int(64 * width_factor)
        self.width = width

        # Initial convolution (accepts 4 MRI modalities)
        self.root = nn.Sequential(
            OrderedDict(
                [
                    ("conv", StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        # Encoder blocks (feature extraction)
        self.body = nn.Sequential(
            OrderedDict(
                [
                    ("block1", self._make_block(width, width * 4, width, block_units[0])),
                    ("block2", self._make_block(width * 4, width * 8, width * 2, block_units[1], stride=2)),
                    ("block3", self._make_block(width * 8, width * 16, width * 4, block_units[2], stride=2)),
                ]
            )
        )

    def _make_block(self, cin, cout, cmid, num_units, stride=1):
        layers = [PreActBottleneck(cin, cout, cmid, stride)]
        for _ in range(1, num_units):
            layers.append(PreActBottleneck(cout, cout, cmid))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        b, _, in_size, _ = x.size()

        x = self.root(x)
        features.append(x)

        # Downsampling
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)

        # Collect intermediate features for skip connections
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            features.append(x)

        x = self.body[-1](x)

        return x, features[::-1]  # reverse features for decoder


class Conv2dReLU(nn.Sequential):
    # Conv → BatchNorm → ReLU block
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)

        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    # Upsampling + skip connection fusion block
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()

        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class SegmentationHead(nn.Sequential):
    # Final segmentation layer (maps features → class logits)
    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsample = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

        super().__init__(conv2d, upsample)


class DecoderCup(nn.Module):
    # Decoder part of UNet (progressive upsampling + skip fusion)
    def __init__(self):
        super().__init__()

        head_channels = 512

        # Reduce encoder output channels
        self.conv_more = Conv2dReLU(1024, head_channels, kernel_size=3, padding=1)

        decoder_channels = (512, 256, 128, 64)
        in_channels = [head_channels] + list(decoder_channels[:-1])

        # Skip connections from encoder
        skip_channels = [512, 256, 64, 0]

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(in_ch, out_ch, sk_ch)
                for in_ch, out_ch, sk_ch in zip(in_channels, decoder_channels, skip_channels)
            ]
        )

    def forward(self, hidden_states, features=None):
        x = self.conv_more(hidden_states)

        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if features is not None and i < len(features) else None
            x = decoder_block(x, skip=skip)

        return x


class ResNetUNet(nn.Module):
    # Complete model: ResNet encoder + UNet decoder
    def __init__(self, num_classes=4, input_channels=4):
        super().__init__()

        self.encoder = ResNetV2(block_units=(3, 4, 9), width_factor=1, in_channels=input_channels)
        self.decoder = DecoderCup()

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        encoder_output, features = self.encoder(x)

        decoder_output = self.decoder(encoder_output, features)

        logits = self.segmentation_head(decoder_output)

        return logits