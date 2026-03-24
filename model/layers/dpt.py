import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .block import resconv

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        # self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        # self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit1 = resconv(features, features, k=3, s=1)
        self.resConfUnit2 = resconv(features, features, k=3, s=1)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.SiLU(inplace=True),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class ProjFeats(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels, 
        lvl=-2,
    ):
        super(ProjFeats, self).__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = []
        for i in range(len(self.projects)):
            if i + lvl < 0:
                self.resize_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=out_channels[i],
                        out_channels=out_channels[i],
                        kernel_size=2**(-i-lvl),
                        stride=2**(-i-lvl),
                        padding=0
                    )
                )
            else:
                self.resize_layers.append(
                    nn.Conv2d(
                        in_channels=out_channels[i],
                        out_channels=out_channels[i],
                        kernel_size=2**(i+lvl),
                        stride=2**(i+lvl),
                        padding=0
                    )
                )
        self.resize_layers = nn.ModuleList(self.resize_layers)
    
    def forward(self, out_features):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        
        return out

class UpsampleFeats(nn.Module):
    def __init__(
        self,
        features, 
        out_channels,
        use_bn=False, 
    ):
        super(UpsampleFeats, self).__init__()
        self.scratch = nn.ModuleList([
            nn.Conv2d(
                in_channels=out_channels[i],
                out_channels=features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ) for i in range(len(out_channels))
        ])

        self.refine = nn.ModuleList([
            _make_fusion_block(features, use_bn, size=None) for _ in range(len(out_channels))
        ])
    
    def forward(self, out):
        out_rn = [self.scratch[i](out[i]) for i in range(len(out))]
        for i in range(1, len(out_rn)+1):
            if i == 1:
                out_rn[-i] = self.refine[-i](out_rn[-i], size=out_rn[-i].shape[2:])
            else:
                up_feat = nn.functional.interpolate(out_rn[-i+1], scale_factor=2, mode='bilinear', align_corners=True)
                out_rn[-i] = self.refine[-i](out_rn[-i], up_feat, size=out_rn[-i].shape[2:])

        return out_rn