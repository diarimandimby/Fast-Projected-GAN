import torch
from torch import nn

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=1)
        )
        self.pointwise = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class SkipLayerExcitation(nn.Module):
  def __init__(self, c_in, c_out):
    super(SkipLayerExcitation, self).__init__()

    self.block = nn.Sequential(
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(c_in, c_out, kernel_size=(4, 4), stride=1, padding=0, bias=False)
        ),
        Swish(),
        nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(c_out, c_out, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        ),
        nn.Sigmoid()
    )

  def forward(self, x_low, x_high):
    return x_high * self.block(x_low)

class FastGANGenerator(nn.Module):
  def __init__(self, output_res=256):
    super(FastGANGenerator, self).__init__()

    multiplier_dict = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}

    self.channels = []
    self.output_res = output_res

    for _, resolution in enumerate(multiplier_dict):
      if(resolution <= self.output_res):
        self.channels.append(int(multiplier_dict[resolution] * 64))

    l = len(self.channels)

    self.initial_block = nn.Sequential(
        nn.utils.parametrizations.spectral_norm(
            nn.ConvTranspose2d(256, self.channels[0]*2, kernel_size=(4, 4), stride=1, padding=0, bias=False)
        ),
        nn.BatchNorm2d(self.channels[0]*2),
        nn.GLU(1)
    )

    self.upblocks = nn.ModuleList()
    for i in range(1, l):
      in_channels = self.channels[i - 1]
      out_channels = self.channels[i]*2
      self.upblocks.append(nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        SeparableConv2d(in_channels, out_channels*2, kernel_size=(3, 3)),
        NoiseInjection(),
        nn.BatchNorm2d(out_channels*2),
        nn.GLU(1),
        SeparableConv2d(out_channels, out_channels, kernel_size=(3, 3)),
        NoiseInjection(),
        nn.BatchNorm2d(out_channels),
        nn.GLU(1)
      ))

    self.skip_layer_excitation = nn.ModuleList()
    for i in range(3):
      c_in = self.channels[i]
      c_out = self.channels[i + 4]
      self.skip_layer_excitation.append(SkipLayerExcitation(c_in, c_out))

    self.final_block = nn.Sequential(
        nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(self.channels[-1], 3, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        ),
        nn.Tanh()
    )

  def normalizeSecondMoment(self, x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

  def forward(self, x):

    x = self.normalizeSecondMoment(x)

    feat4 = self.initial_block(x)

    feat8 = self.upblocks[0](feat4)
    feat16 = self.upblocks[1](feat8)
    feat32 = self.upblocks[2](feat16)

    feat64 = self.skip_layer_excitation[0](feat4, self.upblocks[3](feat32))

    if(self.output_res >= 128):
      feat_last = self.skip_layer_excitation[1](feat8, self.upblocks[4](feat64))

    if(self.output_res >= 256):
      feat_last = self.skip_layer_excitation[2](feat16, self.upblocks[5](feat_last))

    if(self.output_res >= 512):
      feat_last = self.upblocks[6](feat_last)

    if(self.output_res >= 1024):
      feat_last = self.upblocks[7](feat_last)

    return self.final_block(feat_last)
