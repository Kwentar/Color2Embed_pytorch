import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, style_dim=512, bilinear=True):
        # from https://github.com/milesial/Pytorch-UNet
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.G0 = Up(1024, 512 // factor, bilinear)
        self.F0 = ModulatedConv2d(512 // factor, 512 // factor, 3, style_dim)
        self.G1 = Up(512, 256 // factor, bilinear)
        self.F1 = ModulatedConv2d(256 // factor, 256 // factor, 3, style_dim)
        self.G2 = Up(256, 128 // factor, bilinear)
        self.F2 = ModulatedConv2d(128 // factor, 128 // factor, 3, style_dim)
        self.G3 = Up(128, 64, bilinear)
        self.F3 = ModulatedConv2d(64, 64, 3, style_dim)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, color_embed):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.G0(x5, x4)
        x = self.F0(x, color_embed)
        x = self.G1(x, x3)
        x = self.F1(x, color_embed)
        x = self.G2(x, x2)
        x = self.F2(x, color_embed)
        x = self.G3(x, x1)
        # return x
        x = self.F3(x, color_embed)
        logits = self.outc(x)
        return logits


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim
    ):
        # some part from https://github.com/rosinality/stylegan2-pytorch/blob/a2f38914bb5049894c37f2d7a9854bc130cf8a27/model.py
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.weight_linear = nn.Parameter(torch.randn(in_channel, style_dim))
        self.bias_linear = nn.Parameter(torch.zeros(in_channel).fill_(1))
        # self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size})"
        )

    def forward(self, input_, style):
        batch, in_channel, height, width = input_.shape

        # Linear
        style = F.linear(style, self.weight_linear * self.scale, bias=self.bias_linear)

        # Dot
        weight = self.scale * self.weight * style.view(batch, 1, in_channel, 1, 1) # mod weight

        # Norm
        Fnorm = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * Fnorm.view(batch, self.out_channel, 1, 1, 1)

        # Convolve
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        input_ = input_.view(1, batch * in_channel, height, width)
        out = F.conv2d(input_, weight, padding=self.padding,  groups=batch)

        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        return out


class Color2Embed(nn.Module):
    def __init__(self, style_dim):
        super(Color2Embed, self).__init__()
        self.color_encoder = torch_models.resnet18(num_classes=style_dim)
        self.content_encoder_plus_PFFN = UNet(1, 2)

    def forward(self, grayscale_image, color_image):
        z = self.color_encoder(color_image)
        out = self.content_encoder_plus_PFFN(grayscale_image, z)
        return out


if __name__ == '__main__':
    color_2_embed = Color2Embed(512)
    print(color_2_embed(torch.rand(5, 1, 256, 256), torch.rand(5, 3, 256, 256)).shape)

