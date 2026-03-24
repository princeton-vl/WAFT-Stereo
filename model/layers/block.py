import torch.nn as nn

class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, act_layer=nn.SiLU, norm_layer=nn.InstanceNorm2d):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            norm_layer(inp),
            act_layer(inplace=True),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k//2, bias=True),
            norm_layer(oup),
            act_layer(inplace=True),
            nn.Conv2d(oup, oup, kernel_size=k, stride=1, padding=k//2, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)

class conv3x3(nn.Module):
    def __init__(self, inp, oup):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp*2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp*2, oup, 1, padding=0, bias=True)
        )
    def forward(self, x):
        return self.conv(x)