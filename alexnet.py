"""
AlexNet network
  This implementation of AlexNet V2 is mostly same as pytorch (https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)
  For details, please refer to the original paper (
    V1: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    V2: https://arxiv.org/pdf/1404.5997.pdf
  )
"""

import sys
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet_v1', 'alexnet_v1', 'AlexNet_v2', 'alexnet_v2']


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
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


class AlexNet_v1(nn.Module):
  """AlexNet model (version 1)
  Original paper is "ImageNet Classification with Deep Convolutional Neural Networks" (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
  In this implementation, local response normalization layer is replced by batch normalization layer.

  Args:
    num_classes (int): the number of classes
  """
  def __init__(self, num_classes=1000, use_bn=True):
    super(AlexNet, self).__init__()

    self.features = nn.Sequential(
      nn.ConvBlock(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, use_bn=use_bn),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.ConvBlock(in_channels=96, out_channels=256, kernel_size=5, padding=2, use_bn=use_bn),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.ConvBlock(in_channels=256, out_channels=384, kernel_size=3, padding=1, use_bn=use_bn),
      nn.ConvBlock(in_channels=384, out_channels=384, kernel_size=3, padding=1, use_bn=use_bn),
      nn.ConvBlock(in_channels=384, out_channels=256, kernel_size=3, padding=1, use_bn=use_bn)
    )
  
    self.classifer = nn.Sequential(
      nn.Dropout(0.5),
      nn.Lienar(256*6*6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x


class AlexNet_v2(nn.Module):
  """AlexNet model (version 2)
  Original paper is "One Wierd Trick for Parallelizing Convolutional Neural Networks" (https://arxiv.org/pdf/1404.5997.pdf).

  Args:
    num_classes (int): the number of classes
  """
  def __init__(self, num_classes=1000, use_bn=True):
    super(AlexNet_v2, self).__init__()
    self.features = nn.Sequential(
      ConvBlock(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2, use_bn=use_bn),
      nn.MaxPool2d(kernel_size=3, stride=2),
      ConvBlock(in_channels=64, out_channels=192, kernel_size=5, padding=2, use_bn=use_bn),
      nn.MaxPool2d(kernel_size=3, stride=2),
      ConvBlock(in_channels=192, out_channels=384, kernel_size=3, padding=1, use_bn=use_bn),
      ConvBlock(in_channels=384, out_channels=256, kernel_size=3, padding=1, use_bn=use_bn),
      ConvBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, use_bn=use_bn),
      nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x


def alexnet_v1(pretrained=False, num_classes, use_bn=True):
  """AlexNet model (version 1)

  Currently, pre-trained model is not available.
  """

  if pretrained:
    print('pretrained AlexNet v1 is not available.')
    sys.exit()
  
  model = AlexNet_v1(num_classes, use_bn)
  return model


def alexnet_v2(pretrained=False, **kwargs):
  """AlexNet model (version 2)

  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  
  if pretrained:
    print('pretrained AlexNet v2 is not available.')
    sys.exit()

  model = AlexNet_v2(num_classes, use_bn)
  return model
