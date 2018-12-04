import sys
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['LeNet', 'lenet']


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
    super(ConvBlock, self).__init__()

    if use_bn:
      self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
      )
    else:
      self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
      )

  def forward(self, x):
    return self.conv(x)


class LeNet(nn.Module):
  def __init__(self, num_classes, use_bn=True):
    super(LeNet, self).__init__()
    
    self.features = nn.Sequential(
      ConvBlock(in_channels=3, out_channels=6, kernel_size=5),
      nn.MaxPool2d(kernel_size=2),
      ConvBlock(in_channels=6, out_channels=16, kernel_size=5),
      nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
      nn.Linear(16*5*5, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, 84),
      nn.ReLU(inplace=True),
      nn.Linear(84, num_classes)
    )

  def forward(self, x):
    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    return out


def lenet(pretrained=False, num_classes):
  if pretrained:
    print('pretrained LeNet is not available.')
    sys.exit()

  model = LeNet(num_classes, use_bn)
  return model