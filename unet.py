"""
U-Net
  This implementation is partially same as milesial (https://github.com/milesial/Pytorch-UNet/tree/master/unet)
  For details, please refer to the original paper (https://arxiv.org/pdf/1505.04597.pdf)
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, batch_norm=True):
    super(DoubleConv, self).__init__()
    if batch_norm:
      self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
      )
    else:
      self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(True)
      )

  def forward(self, x):
    x = self.conv(x)
    return x

class InBlock(nn.Module):
  def __init__(self, in_channels, out_channels, batch_norm=True):
    super(InBlock, self).__init__()
    self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm)

  def forward(self, x):
    x = self.conv(x)
    return x

class DownBlock(nn.Module):
  def __init__(self, in_channels, out_channels, batch_norm=True):
    super(DownBlock, self).__init__()
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm)
  
  def forward(self, x):
    x = self.maxpool(x)
    x = self.conv(x)
    return x

class UpBlock(nn.Module):
  def __init__(self, in_channels, out_channels, bilinear=True, batch_norm=True):
    super(UpBlock, self).__init__()
    if bilinear:
      self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    else:
      self.upscale = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
    
    self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm)
  
  def forward(self, from_down, from_up):
    from_up = self.upscale(from_up)
    x = torch.cat([from_down, from_up], dim=1)
    x = self.conv(x)
    return x

class OutBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
  
  def forward(self, x):
    x = self.conv(x)
    return x

class UNet(nn.Module):
  def __init__(self, num_channels, num_classes):
    super(UNet, self).__init__()
    self.in_block = InBlock(num_channels, 64)
    self.down_block1 = DownBlock(64, 128)
    self.down_block2 = DownBlock(128, 256)
    self.down_block3 = DownBlock(256, 512)
    self.down_block4 = DownBlock(512, 512)
    self.up_block1 = UpBlock(1024, 256)
    self.up_block2 = UpBlock(512, 128)
    self.up_block3 = UpBlock(256, 64)
    self.up_block4 = UpBlock(128, 64)
    self.out_block = OutBlock(64, num_channels)
  
  def forward(self, x):
    x1 = self.in_block(x)
    x2 = self.down_block1(x1)
    x3 = self.down_block2(x2)
    x4 = self.down_block3(x3)
    x5 = self.down_block4(x4)
    x = self.up_block1(x4, x5)
    x = self.up_block2(x3, x)
    x = self.up_block3(x2, x)
    x = self.up_block4(x1, x)
    x = self.out_block(x)
    return x
