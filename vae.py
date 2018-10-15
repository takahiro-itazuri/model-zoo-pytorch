"""
VAE (Variational Auto-Encoder)
  This implementation is completely same as pytorch example (https://github.com/pytorch/examples/tree/master/vae).
  For details, please refer to the original paper (https://arxiv.org/pdf/1312.6114.pdf)
"""

import torch.nn as nn
import torch.nn.functional as F

"""
Hyperparameters
  nz (int): the size of latent vector (z)
"""

class VAE(nn.Module):
  """
  VAE class
    This class has an encoder and a decoder.
    Both has a single hidden layer (400 neurons).

  Model:
    fc1: 784 -> 400
    fc21: 400 -> 20
    fc22: 400 -> 20
    fc3: 20 -> 400
    fc4: 400 -> 784
  """
  def __init__(self):
    super(VAE, self).__init__()

    self.fc1 = nn.Linear(784, 400)
    self.fc21 = nn.Linear(400, 20)
    self.fc22 = nn.Linear(400, 20)
    self.fc3 = nn.Linear(20, 400)
    self.fc4 = nn.Linear(400, 784)

  def encode(self, x):
    x = F.relu(self.fc1(x))
    mu = self.fc21(x)
    logvar = self.fc22(x)
    return mu, logvar
  
  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

  def decode(self, z):
    z = F.relu(self.fc3(z))
    z = torch.sigmoid(self.fc4(z))
    return z

  def forward(self, x)
    mu, logvar = self.encode(x.view(-1, 784))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar