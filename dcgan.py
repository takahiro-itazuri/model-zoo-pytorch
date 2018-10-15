"""
DCGAN
  This implementation is completely same as pytorch examples (https://github.com/pytorch/examples/tree/master/dcgan).
  For details, please refer to the original paper (https://arxiv.org/pdf/1511.06434.pdf).
"""

import torch.nn as nn

"""
Hyperparameters
  nc (int): channel size of image
  nz (int): size of latent vector (z)
  ngf (int): parameter for generator filter size
  ndf (int): parameter for discriminator filter size
"""
nc = 3       
nz = 100     
ngf = 128    
ndf = 128    

class Generator(nn.Module):
  """
  Generator class
  
  Model:
    deconv1: (nz, 1, 1) -> (ngf*8, 4, 4)
    deconv2: (ngf*8, 4, 4) -> (ngf*4, 8, 8)
    deconv3: (ngf*4, 8, 8) -> (ngf*2, 16, 16)
    deconv4: (ngf*2, 16, 16) -> (ngf, 32, 32)
    decvon5: (ngf, 32, 32) -> (nc, 64, 64)
  """
  def __init__(self):
    super(Generator, self).__init__()

    self.deconv1 = nn.Sequential(
      nn.ConvTranspose2d(nz, ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(ngf*8),
      nn.ReLU(inplace=True)
    )

    self.deconv2 = nn.Sequential(
      nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ngf*4),
      nn.ReLU(inplace=True)
    )

    self.deconv3 = nn.Sequential(
      nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ngf*2),
      nn.ReLU(inplace=True)
    )

    self.deconv4 = nn.Sequential(
      nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(inplace=True)
    )

    self.deconv5 = nn.Sequential(
      nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
      nn.Tanh()
    )

  def forward(self, x):
    x = self.deconv1(x)
    x = self.deconv2(x)
    x = self.deconv3(x)
    x = self.deconv4(x)
    x = self.deconv5(x)
    return x

class Discriminator(nn.Module):
  """
  Discriminator class
  
  Model:
    conv1: (nc, 64, 64) -> (ngf, 32, 32)
    conv2: (ngf, 32, 32) -> (ngf*2, 16, 16)
    conv3: (ngf*2, 16, 16) -> (ngf*4, 8, 8)
    conv4: (ngf*4, 8, 8) -> (ngf*8, 4, 4)
    cvon5: (ngf*8, 4, 4) -> (1, 1, 1)
  """
  def __init__(self):
    self.conv1 = nn.Sequential(
      nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.2, inplace=True)
    )
    
    self.conv2 = nn.Sequential(
      nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ndf*2),
      nn.LeakyReLU(0.2, inplace=True)
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ndf*4),
      nn.LeakyReLU(0.2, inplace=True)
    )

    self.conv4 = nn.Sequential(
      nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ndf*8),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.conv5 = nn.Sequential(
      nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
      nn.Sigmoid()
    )

  def forward(self, input):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    return x.view(-1, 1).squeeze(1)