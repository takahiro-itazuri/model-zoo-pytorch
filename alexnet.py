"""
AlexNet network
  This implementation of AlexNet V2 is completely same as pytorch (https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)
  For details, please refer to the original paper (
    V1: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    V2: https://arxiv.org/pdf/1404.5997.pdf
  )
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet_v1', 'alexnet_v1', 'AlexNet_v2', 'alexnet_v2']

model_urls = {
  'alexnet_v2': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}

class AlexNet_v1(nn.Module):
  """AlexNet model (version 1)
  Original paper is "ImageNet Classification with Deep Convolutional Neural Networks" (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
  In this implementation, local response normalization layer is replced by batch normalization layer.

  Args:
    num_classes (int): the number of classes
  """
  def __init__(self, num_classes=1000):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.BatchNorm2d(96),
      nn.Conv2d(96, 256, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.BatchNorm2d(256),
      nn.Conv2d(256, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(256)
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
    x = self.features()
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x


class AlexNet_v2(nn.Module):
  """AlexNet model (version 2)
  Original paper is "One Wierd Trick for Parallelizing Convolutional Neural Networks" (https://arxiv.org/pdf/1404.5997.pdf).

  Args:
    num_classes (int): the number of classes
  """
  def __init__(self, num_classes=1000):
    super(AlexNet_v2, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
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
    x = self.features()
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x


def alexnet_v1(**kwargs):
  """AlexNet model (version 1)

  Currently, pre-trained model is not available.
  """
  model = AlexNet_v1(**kwargs)
  return model


def alexnet_v2(pretrained=False, **kwargs):
  """AlexNet model (version 2)

  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = AlexNet_v2(**kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
  return model
