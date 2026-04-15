"""
Vanilla U-Net — training-from-scratch baseline.

Included as a controlled comparison against the pretrained encoder variant
(unet_pretrained.py) to quantify the value of transfer learning on the
small EWS dataset.

Reference:
    Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
    Image Segmentation." MICCAI 2015. https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Vanilla U-Net for binary segmentation (training from scratch).

    Args:
        in_channels:  RGB input channels (3).
        out_channels: Output classes (1 for binary).
        features:     Encoder channel progression.
        dropout:      Spatial dropout rate in DoubleConv blocks.
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512], dropout=0.1):
        super().__init__()
        self.encoders   = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f, dropout))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout)

        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.decoders.append(DecoderBlock(ch, f, dropout))
            ch = f

        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)


if __name__ == "__main__":
    m = UNet()
    x = torch.randn(2, 3, 350, 350)
    print(f"Output: {m(x).shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
