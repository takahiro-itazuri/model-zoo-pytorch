"""
U-Net
  This implementation is partially same as jaxony (https://github.com/jaxony/unet-pytorch/blob/master/model.py)
  For details, please refer to the original paper (https://arxiv.org/pdf/1505.04597.pdf)
"""

import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def upconv2x2(in_channels, out_channels, mode='transpose'):
  if mode == 'transpose':
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
  else:
    return nn.Sequential(
      nn.Upsample(mode='bilinear', scale_factor=2),
      conv1x1(in_channels, out_channels)
    )

def conv1x1(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

class DownConv(nn.Module):
  def __init__(self, in_channels, out_channels, pooling=True, batch_norm=True):
    super(DownConv, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.pooling = pooling
    self.batch_norm

    if self.batch_norm:
      self.conv = nn.Sequential(
        conv3x3(self.in_channels, self.out_channels),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True),
        conv3x3(self.out_channels, self.out_channels),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True)
      )
    else:
      self.conv = nn.Sequential(
        conv3x3(self.in_channels, self.out_channels),
        nn.ReLU(True),
        conv3x3(self.out_channels, self.out_channels),
        nn.ReLU(True)
      )

    if self.pooling:
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
  
  def forward(self, x):
    x = conv(x)
    before_pool = x
    if self.pooling:
      x = self.pool(x)
    return x, before_pool

def UpConv(nn.Module):
  def __init__(self, in_channels, out_channels, up_mode='transpose', batch_norm=True):
    super(UpConv, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.up_mode = up_mode
    self.batch_norm = batch_norm

    self.upconv = upconv2x2(self.in_channels, out_channels, mode=self.up_mode)

    if self.batch_norm:
      self.conv = nn.Sequential(
        conv3x3(2*self.out_channels, self.out_channels),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True),
        conv3x3(self.out_channels, self.out_channels),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True)
      )
    else:
      self.conv = nn.Sequential(
        conv3x3(2*self.out_channels, self.out_channels),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True),
        conv3x3(self.out_channels, self.out_channels),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True)
      )
  
  def forward(self, from_down, from_up):
    """Forward pass
    Args:
      from_down: tensor from the encoder
      from_up: tensor from the decoder
    """
    from_up = self.upconv(from_up)
    x = torch.cat((from_up, from_down), 1)
    x = self.conv(x)
    return x
  
def UNet(nn.Module):
  def __init__(self, num_classes, in_channels=3, depth=5, up_mode='transpose', batch_norm=True):
    if up_mode in ('transpose', 'upsample'):
      self.up_mode = up_mode
    else:
      raise ValueError('"{}" is not a valid mode for upsampling. Only "transpose" and "upsample" are allowed.'.format(up_mode))
    
    self.num_classes = num_classes
    self.in_channels = in_channels
    self.depth = depth
    self.batch_norm = batch_norm

    self.down_convs = []
    self.up_convs = []

    for i in range(depth):
      ins = self.in_channels if i == 0 else outs
      outs = 64 * (2**i)
      pooling = True if i < depth-1 else False
      self.down_convs.append(DownConv(ins, outs, pooling=pooling, batch_norm=self.batch_norm))

    for i in range(depth-1):
      ins = outs
      outs = ins // 2
      self.up_convs.append(UpConv(ins, outs, up_mode=self.up_mode, batch_norm=self.batch_norm))
    
    self.conv_final = conv1x1(outs, self.num_classes)

    self._initialize_weights()
  
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
  
  def forward(self, x):
    encoder_outs = []

    for i, down_conv in enumerate(self.down_convs):
      x, before_pool = down_conv(x)
      if i < self.depth - 1:
        encoder_outs.append(before_pool)
    
    for i, up_conv in enumerate(self.up_convs):
      before_pool = encoder_outs[self.depth - 2 - i]
      x = up_conv(before_pool, x)
    
    x = self.conv_final(x)
    return x
