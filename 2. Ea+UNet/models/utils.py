import torch
import torch.nn as nn
import torch.nn.functional as F

def np2th(weights, conv=False):
    """Convert numpy weights to torch tensor (HWIO → OIHW for conv layers)."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class StdConv2d(nn.Conv2d):
    """
    Weight-standardized convolution.
    Before each forward pass, the filter weights are normalized to have
    zero mean and unit variance per output channel. This improves gradient
    flow and training stability, especially with GroupNorm.
    """
    def forward(self, x):
        w = self.weight
        # Compute per-filter mean and variance (dims = [in_ch, kH, kW])
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    """3×3 weight-standardized conv with same-size padding."""
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    """1×1 weight-standardized conv for channel projection."""
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)
