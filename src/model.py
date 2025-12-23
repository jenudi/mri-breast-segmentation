import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv => BN => ReLU => Dropout) * 2"""

    def __init__(self, in_channels, out_channels, dropout_p: float = 0.0):
        super().__init__()
        drop = nn.Dropout2d(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            drop,

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            drop,
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling: maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout_p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upscaling: upsample then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p: float = 0.0):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

        self.bilinear = bilinear

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        bilinear=True,
        dropout_p: float = 0.1,          # default dropout in most blocks
        bottleneck_dropout_p: float = 0.3 # stronger dropout at the bottom
    ):
        super().__init__()

        self.inc   = DoubleConv(n_channels, 64, dropout_p=dropout_p)
        self.down1 = Down(64, 128, dropout_p=dropout_p)
        self.down2 = Down(128, 256, dropout_p=dropout_p)
        self.down3 = Down(256, 512, dropout_p=dropout_p)
        self.down4 = Down(512, 512, dropout_p=bottleneck_dropout_p)  # bottom

        self.up1   = Up(512 + 512, 256, bilinear, dropout_p=dropout_p)
        self.up2   = Up(256 + 256, 128, bilinear, dropout_p=dropout_p)
        self.up3   = Up(128 + 128, 64,  bilinear, dropout_p=dropout_p)
        self.up4   = Up(64 + 64,   64,  bilinear, dropout_p=dropout_p)

        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)
