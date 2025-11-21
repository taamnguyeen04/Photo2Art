import torch
import torch.nn as nn
import numpy as np


class InstanceNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x):
        return self.norm(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, apply_instancenorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False)
        ]
        if apply_instancenorm:
            layers.append(InstanceNorm(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, apply_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False),
            InstanceNorm(out_channels)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, conv_dim=64, repeat_num=8):
        super(Generator, self).__init__()

        # Encoder (Downsampling)
        self.down1 = Downsample(input_channels, 64, apply_instancenorm=False)  # 128x128
        self.down2 = Downsample(64, 128)   # 64x64
        self.down3 = Downsample(128, 256)  # 32x32
        self.down4 = Downsample(256, 512)  # 16x16
        self.down5 = Downsample(512, 512)  # 8x8
        self.down6 = Downsample(512, 512)  # 4x4
        self.down7 = Downsample(512, 512)  # 2x2
        self.down8 = Downsample(512, 512, apply_instancenorm=False)  # 1x1

        # Decoder (Upsampling)
        self.up1 = Upsample(512, 512, apply_dropout=True)   # 2x2
        self.up2 = Upsample(1024, 512, apply_dropout=True)  # 4x4
        self.up3 = Upsample(1024, 512, apply_dropout=True)  # 8x8
        self.up4 = Upsample(1024, 512)  # 16x16
        self.up5 = Upsample(1024, 256)  # 32x32
        self.up6 = Upsample(512, 128)   # 64x64
        self.up7 = Upsample(256, 64)    # 128x128

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        output = self.final(torch.cat([u7, d1], dim=1))
        return output


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, image_size=224, conv_dim=64, c_dim=17, repeat_num=8):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: 128x128
            Downsample(input_channels, 64, apply_instancenorm=False),

            # Layer 2: 64x64
            Downsample(64, 128),

            # Layer 3: 32x32
            Downsample(128, 256),

            # Layer 4: 34x34 -> 31x31
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            InstanceNorm(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 33x33 -> 30x30
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, stride=1)
        )

    def forward(self, x):
        return self.model(x)
