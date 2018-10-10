import torch
from torch import nn

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      m.weight.data.normal_(0, 0.02)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
      m.weight.data.normal_(0, 0.02)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.02)
      if m.bias is not None:
        m.bias.data.zero_()

class Encoder(nn.Module):
  def __init__(self, latent_dims):
    super(Encoder, self).__init__()

    # input params
    self.latent_dims = latent_dims

    # conv1: 3 x 64 x 64 -> 64 x 32 x 32
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    # conv2: 64 x 32 x 32 -> 128 x 16 x 16
    self.conv2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    # conv3: 128 x 16 x 16 -> 256 x 8 x 8
    self.conv3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    # fc4: 256 * 8 * 8 -> 2048
    self.fc4 = nn.Sequential(
      nn.Linear(256 * 8 * 8, 2048, bias=False),
      nn.BatchNorm1d(2048),
      nn.ReLU(inplace=True)
    )
  
    # mu: 2048 -> latent_dims
    self.mu = nn.Linear(2048, latent_dims, bias=False)

    # logvar: 2048 -> latent_dims
    self.logvar = nn.Linear(2048, latent_dims, bias=False)

    initialize_weights(self)
  
  def forward(self, x):
    batch_size = x.size(0)

    h = self.conv1(x)
    h = self.conv2(h)
    h = self.conv3(h)
    h = h.view(batch_size, -1)
    h = self.fc4(h)
    mu = self.mu(h)
    logvar = self.logvar(h)
    return mu, logvar

  
class Decoder(nn.Module):
  def __init__(self, latent_dims):
    super(Decoder, self).__init__()

    # input params
    self.latent_dims = latent_dims

    # fc1: latent_dims -> 256 * 8 * 8
    self.fc1 = nn.Sequential(
      nn.Linear(latent_dims, 256 * 8 * 8, bias=False),
      nn.BatchNorm1d(256 * 8 * 8),
      nn.ReLU(inplace=True)
    )

    # deconv2: 256 x 8 x 8 -> 256 x 16 x 16
    self.deconv2 = nn.Sequential(
      nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    # deconv3: 256 x 16 x 16 -> 128 x 32 x 32
    self.deconv3 = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    # deconv4: 128 x 32 x 32 -> 32 x 64 x 64
    self.deconv4 = nn.Sequential(
      nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True)
    )

    # deconv5: 32 x 64 x 64 -> 3 x 64 x 64
    self.conv5 = nn.Sequential(
      nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2, bias=False),
      nn.Tanh()
    )

    initialize_weights(self)

  def forward(self, z):
    xhat = self.fc1(z)
    xhat = xhat.view(-1, 256, 8, 8)
    xhat = self.deconv2(xhat)
    xhat = self.deconv3(xhat)
    xhat = self.deconv4(xhat)
    xhat = self.conv5(xhat)
    return xhat

class Generator(nn.Module):
  def __init__(self, latent_dims):
    super(Generator, self).__init__()

    # input params
    self.latent_dims = latent_dims

    self.encoder = Encoder(latent_dims)
    self.decoder = Decoder(latent_dims)
  
  def reparameterize(self, mu, logvar):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu

  def forward(self, x):
    mu, logvar = self.encoder(x)
    z = self.reparameterize(mu, logvar)
    xhat = self.decoder(z)
    return xhat, mu, logvar
  
  def generate(self, z):
    self.eval()
    samples = self.decoder(z)
    return samples

  def reconstruct(self, x):
    self.eval()
    mu, _ = self.encoder(x)
    xhat = self.decoder(mu)

    return xhat
  

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    # conv1: 3 x 64 x 64 -> 32 x 64 x 64
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
      nn.ReLU(inplace=True)
    )

    # conv2: 32 x 64 x 64 -> 128 x 32 x 32
    self.conv2 = nn.Sequential(
      nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    # conv3: 128 x 32 x 32 -> 256 x 16 x 16
    self.conv3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    # conv4: 256 x 16 x 16 -> 256 x 8 x 8
    self.conv4 = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    # fc5: 256 * 8 * 8 -> 512
    self.fc5 = nn.Sequential(
      nn.Linear(256 * 8 * 8, 512, bias=False),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True)
    )

    # fc6: 512 -> 1
    self.fc6 = nn.Sequential(
      nn.Linear(512, 1),
    )

    initialize_weights(self)

  def forward(self, x):
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    f = self.conv4(f)
    f = f.view(-1, 256 * 8 * 8)
    f = self.fc5(f)
    o = self.fc6(f)
    return o
