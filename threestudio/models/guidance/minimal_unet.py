import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = output_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.downsample_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(input_channels, output_channels)
        )

    def forward(self, x):
        return self.downsample_conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_block = ConvBlock(input_channels, output_channels, input_channels // 2)
        else:
            self.upsample = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)
            self.conv_block = ConvBlock(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


class OutputConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MinimalUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, bilinear=False):
        super(MinimalUNet, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.bilinear = bilinear

        self.initial_conv = ConvBlock(num_input_channels, 64)
        self.downsample1 = DownsampleBlock(64, 128)
        self.downsample2 = DownsampleBlock(128, 256)
        self.downsample3 = DownsampleBlock(256, 512)
        factor = 2 if bilinear else 1
        self.downsample4 = DownsampleBlock(512, 1024 // factor)
        self.upsample1 = UpsampleBlock(1024, 512 // factor, bilinear)
        self.upsample2 = UpsampleBlock(512, 256 // factor, bilinear)
        self.upsample3 = UpsampleBlock(256, 128 // factor, bilinear)
        self.upsample4 = UpsampleBlock(128, 64, bilinear)
        self.output_conv = OutputConv(64, num_output_channels)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x5 = self.downsample4(x4)
        x = self.upsample1(x5, x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x, x2)
        x = self.upsample4(x, x1)
        logits = self.output_conv(x)
        return logits

    def use_checkpointing(self):
        self.initial_conv = torch.utils.checkpoint(self.initial_conv)
        self.downsample1 = torch.utils.checkpoint(self.downsample1)
        self.downsample2 = torch.utils.checkpoint(self.downsample2)
        self.downsample3 = torch.utils.checkpoint(self.downsample3)
        self.downsample4 = torch.utils.checkpoint(self.downsample4)
        self.upsample1 = torch.utils.checkpoint(self.upsample1)
        self.upsample2 = torch.utils.checkpoint(self.upsample2)
        self.upsample3 = torch.utils.checkpoint(self.upsample3)
        self.upsample4 = torch.utils.checkpoint(self.upsample4)
        self.output_conv = torch.utils.checkpoint(self.output_conv)
