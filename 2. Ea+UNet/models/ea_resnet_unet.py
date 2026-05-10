import torch.nn as nn
from models.encoder import ResNetV2
from models.decoder import DecoderCup, SegmentationHead

class ResNetUNet(nn.Module):
    """
    Edge-Attention ResNet-UNet v2.
    """
    def __init__(self, num_classes=4, input_channels=4):
        super().__init__()

        # Encoder: ResNet v2 with parallel EdgeAttentionBranch embedded
        self.encoder = ResNetV2(
            block_units=(3, 4, 9),   # block depths (ResNet50-like)
            width_factor=1,           # base channels = 64
            in_channels=input_channels
        )

        # Decoder: 4-stage UNet with skip connections + edge attention
        self.decoder = DecoderCup()

        # Segmentation head: 64ch → num_classes logits (1×1 conv)
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        """
        Returns:
            logits      : Segmentation class logits [B, num_classes, H, W]
            lateral_edge: Raw edge logit [B, 1, H, W]
        """
        encoder_output, features, lateral_edge = self.encoder(x)

        # Decoder receives lateral_edge and applies it as an attention map
        decoder_output = self.decoder(encoder_output, features, lateral_edge)

        logits = self.segmentation_head(decoder_output)

        return logits, lateral_edge
