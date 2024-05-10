
#Unet with SE in Skip conn. 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Make sure to flatten to (batch_size, channels)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class UpSkipSE(nn.Module):
    """Upscaling then double conv with optional SE block in skip connection"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_se=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        self.use_se = use_se
        if self.use_se:
            self.se_block = SEBlock(out_channels)  # Apply SE block to the concatenated features

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_se:
            x = self.se_block(x)
        return x

    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetSESkip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetSESkip, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, dilation=1)
        self.down2 = Down(128, 256, dilation=2)
        self.down3 = Down(256, 512, dilation=4)
        self.down4 = Down(512, 512, dilation=8)
        self.up1 = UpSkipSE(1024, 256, bilinear, use_se=True)  # Apply SE block in skip connection
        self.up2 = UpSkipSE(512, 128, bilinear, use_se=True)   # Apply SE block in skip connection
        self.up3 = UpSkipSE(256, 64, bilinear, use_se=True)    # Apply SE block in skip connection
        self.up4 = UpSkipSE(128, 64, bilinear, use_se=True)    # Apply SE block in skip connection
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # print("Input:", x.shape)
        x1 = self.inc(x)
        # print("After inc:", x1.shape)
        x2 = self.down1(x1)
        # print("After down1:", x2.shape)
        x3 = self.down2(x2)
        # print("After down2:", x3.shape)
        x4 = self.down3(x3)
        # print("After down3:", x4.shape)
        x5 = self.down4(x4)
        # print("After down4:", x5.shape)
        x = self.up1(x5, x4)
        # print("After up1:", x.shape)
        x = self.up2(x, x3)
        # print("After up2:", x.shape)
        x = self.up3(x, x2)
        # print("After up3:", x.shape)
        x = self.up4(x, x1)
        # print("After up4:", x.shape)
        logits = self.outc(x)
        # print("Output:", logits.shape)
        return logits

