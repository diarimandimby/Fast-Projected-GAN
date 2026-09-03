import torch
from torch import nn
import timm
from DiffAugment_pytorch import DiffAugment
import torch.nn.functional as F

class DownBlock(nn.Module):
  def __init__(self, c_in, c_out):
    super(DownBlock, self).__init__()
    self.block = nn.Sequential(
        nn.utils.parametrizations.spectral_norm(
          nn.Conv2d(c_in, c_out, kernel_size=(4, 4), stride=2, padding=1)
        ),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2, inplace=True)
    )

  def forward(self, input):
    return self.block(input)

class DiscriminatorL1(nn.Module):
  def __init__(self, c1):
    super(DiscriminatorL1, self).__init__()
    self.block = nn.Sequential(
        DownBlock(c1, 64),
        DownBlock(64, 128),
        DownBlock(128, 256),
        DownBlock(256, 512),
        nn.utils.parametrizations.spectral_norm(
          nn.Conv2d(512, 1, kernel_size = (4, 4), stride=2, padding=1)
        )
    )

  def forward(self, input):
    return self.block(input)

class DiscriminatorL2(nn.Module):
  def __init__(self, c2):
    super(DiscriminatorL2, self).__init__()
    self.block = nn.Sequential(
        DownBlock(c2, 128),
        DownBlock(128, 256),
        DownBlock(256, 512),
        nn.utils.parametrizations.spectral_norm(
          nn.Conv2d(512, 1, kernel_size = (4, 4), stride=2, padding=1)
        )
    )

  def forward(self, input):
    return self.block(input)

class DiscriminatorL3(nn.Module):
  def __init__(self, c3):
    super(DiscriminatorL3, self).__init__()
    self.block = nn.Sequential(
        DownBlock(c3, 256),
        DownBlock(256, 512),
        nn.utils.parametrizations.spectral_norm(
          nn.Conv2d(512, 1, kernel_size = (4, 4), stride=2, padding=1)
        )
    )

  def forward(self, input):
    return self.block(input)

class DiscriminatorL4(nn.Module):
  def __init__(self, c4):
    super(DiscriminatorL4, self).__init__()
    self.block = nn.Sequential(
        DownBlock(c4, 512),
        nn.utils.parametrizations.spectral_norm(
          nn.Conv2d(512, 1, kernel_size = (4, 4), stride=2, padding=1)
        )
    )

  def forward(self, input):
    return self.block(input)

class CCM(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(CCM, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_in', nonlinearity='linear')

    for param in self.conv.parameters():
      param.requires_grad = False

  def forward(self, input):
    return self.conv(input)

class CSM(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(CSM, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)

    self.skip_add = nn.quantized.FloatFunctional()

    torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    for param in self.conv.parameters():
      param.requires_grad = False

  def forward(self, x, y):
    output = x

    if y is not None:
      output = self.skip_add.add(output, y)

    output = nn.functional.interpolate(
        output, scale_factor=2, mode="bilinear", align_corners=True
    )

    return self.conv(output)

class ProjectedGANDiscriminator(nn.Module):
  def __init__(self):
    super(ProjectedGANDiscriminator, self).__init__()

    self.feature_extractor = timm.create_model(
        'tf_efficientnet_lite0.in1k',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4)
        )

    in_channels = [24, 40, 112, 320]

    out_channels = [64, 128, 256, 512]

    self.ccm = nn.ModuleList()
    for i in range(4):
      self.ccm.append(CCM(in_channels[i], out_channels[i]))

    self.csm = nn.ModuleList()
    for i in range(4):
      self.csm.append(CSM(out_channels[3 - i], out_channels[3 - i]//2))

    self.discriminators = nn.ModuleList([
        DiscriminatorL1(out_channels[0]//2),
        DiscriminatorL2(out_channels[1]//2),
        DiscriminatorL3(out_channels[2]//2),
        DiscriminatorL4(out_channels[3]//2)
    ])

  def forward(self, x, diffaug=True, interpolate=True):

    if(diffaug):
      x = DiffAugment(x, policy='color,translation,cutout')

    if(interpolate):
      x = F.interpolate(x, 256, mode='bilinear', align_corners=False)

    with torch.no_grad():
      features = self.feature_extractor(x)
      
    x = None

    logits = []

    for i in range(4):

      x = self.csm[i](self.ccm[3 - i](features[3 - i]), x)

      logits.append(self.discriminators[3 - i](x))

    logits = torch.cat(logits, dim=1)

    return logits
