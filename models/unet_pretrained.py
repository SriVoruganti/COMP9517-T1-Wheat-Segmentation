"""
U-Net with a pretrained ResNet-34 encoder.

Using a pretrained backbone dramatically improves performance on small datasets
like EWS (190 images) by leveraging ImageNet features instead of learning
low-level features from scratch.

The decoder is a lightweight upsampling head with skip connections
from the ResNet-34 residual stages.

Reference:
    He et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
    Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
    Image Segmentation." MICCAI 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample → concat skip → ConvBnRelu."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PretrainedUNet(nn.Module):
    """
    U-Net with ResNet-34 encoder pretrained on ImageNet.

    Encoder stages and their output channels:
        stem  (conv1 + bn + relu):  64  ch, /2
        layer1 (res block 1):       64  ch, /4
        layer2 (res block 2):       128 ch, /8
        layer3 (res block 3):       256 ch, /16
        layer4 (res block 4):       512 ch, /32

    Args:
        out_channels:  Number of output segmentation classes (1 for binary).
        pretrained:    If True, load ImageNet weights for encoder.
        freeze_encoder: If True, freeze encoder weights during early training.
    """

    def __init__(
        self,
        out_channels:   int  = 1,
        pretrained:     bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        # ---- Encoder: ResNet-34 backbone ----
        weights  = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # → 64,  H/2
        self.pool = backbone.maxpool                                              # → 64,  H/4
        self.enc1 = backbone.layer1   # → 64,  H/4
        self.enc2 = backbone.layer2   # → 128, H/8
        self.enc3 = backbone.layer3   # → 256, H/16
        self.enc4 = backbone.layer4   # → 512, H/32

        if freeze_encoder:
            for param in [*self.enc0.parameters(), *self.enc1.parameters(),
                          *self.enc2.parameters()]:
                param.requires_grad = False

        # ---- Decoder ----
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.dec1 = DecoderBlock(64,  64,  32)

        # Final upsample back to input resolution
        self.final_up   = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBnRelu(16, 16),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e0 = self.enc0(x)           # 64,  H/2
        e1 = self.enc1(self.pool(e0))  # 64,  H/4
        e2 = self.enc2(e1)          # 128, H/8
        e3 = self.enc3(e2)          # 256, H/16
        e4 = self.enc4(e3)          # 512, H/32

        # Decode with skip connections
        d  = self.dec4(e4, e3)      # 256, H/16
        d  = self.dec3(d,  e2)      # 128, H/8
        d  = self.dec2(d,  e1)      # 64,  H/4
        d  = self.dec1(d,  e0)      # 32,  H/2

        d  = self.final_up(d)       # 16,  H
        return self.final_conv(d)   # out_channels, H

    def unfreeze_encoder(self):
        """Call after initial training to fine-tune the full network."""
        for param in self.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = PretrainedUNet(pretrained=False)   # pretrained=False for offline check
    dummy = torch.randn(2, 3, 350, 350)
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")              # Expected: (2, 1, 350, 350)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
