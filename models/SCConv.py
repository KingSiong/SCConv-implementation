import torch
from torch import nn
import torch.nn.functional as F

class SRU(nn.Module):
    def __init__(self, 
                 num_groups,
                 num_channels,
                 eps=1e-5,
                 affine=True,
                 device=None,
                 dtype=None,
                 threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.gn = nn.GroupNorm(num_groups, num_channels, eps, affine, device, dtype)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        if x.shape[1] <= 1:
            return x
        # separate
        y = self.gn(x)
        normal_weight = (self.gn.weight / sum(self.gn.weight)).view(-1, 1, 1)
        normal_weight = self.sigmoid(y * normal_weight)

        mask_1 = normal_weight >= self.threshold
        mask_2 = normal_weight < self.threshold

        # reconstruct
        x_1 = x * mask_1
        x_2 = x * mask_2

        x_1_split = torch.split(x_1, x_1.shape[1] // 2, dim=1)
        x_2_split = torch.split(x_2, x_2.shape[1] // 2, dim=1)

        concat_list = [x_1_split[0] + x_2_split[1], x_1_split[1] + x_2_split[0]]
        if x.shape[1] % 2 == 1:
            concat_list.append(x_1_split[2] + x_2_split[2])

        return torch.cat(concat_list, dim=1)

class CRU(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 kernel_size, 
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 alpha=0.5, 
                 squeeze_ratio=2):
        super().__init__()
        # init
        self.up_channels = int(alpha * in_channels)
        self.low_channels = in_channels - self.up_channels
        self.up_squeeze_channels = max(self.up_channels // squeeze_ratio, 1)
        self.low_squeeze_channels = max(self.low_channels // squeeze_ratio, 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # split block
        self.up_squeeze_conv = nn.Conv2d(self.up_channels, self.up_squeeze_channels, kernel_size=1)
        self.low_squeeze_conv = nn.Conv2d(self.low_channels, self.low_squeeze_channels, kernel_size=1)

        # transform block
        self.gwc = nn.Conv2d(
            self.up_squeeze_channels, 
            out_channels, 
            kernel_size,
            stride=stride, 
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)
        self.up_pwc = nn.Conv2d(self.up_squeeze_channels, out_channels, kernel_size=1)
        self.low_pwc = nn.Conv2d(self.low_squeeze_channels, out_channels - self.low_squeeze_channels, kernel_size=1)

        # fuse block
        self.pool = nn.AdaptiveAvgPool2d(1)

        # in_channels = 1
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size,
            stride=stride, 
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)

    def forward(self, x):
        if self.up_channels == 0:
            return self.conv(x)
            
        # split
        x_up, x_low = torch.split(x, [self.up_channels, self.low_channels], dim=1)
        x_up = self.up_squeeze_conv(x_up)
        x_low = self.low_squeeze_conv(x_low)

        # transform
        h_in = x.shape[2]
        w_in = x.shape[3]
        h_out = (h_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        y_1 = self.gwc(x_up) + F.interpolate(self.up_pwc(x_up), size=(h_out, w_out), mode='nearest')
        y_2 = F.interpolate(torch.cat([self.low_pwc(x_low), x_low], dim=1), size=(h_out, w_out), mode='nearest')

        # fuse
        s_1 = self.pool(y_1)
        s_2 = self.pool(y_2)
        beta = F.softmax(torch.cat([s_1, s_2], dim=2), dim=2)
        beta_1, beta_2 = torch.split(beta, beta.shape[2] // 2, dim=2)
        return beta_1 * y_1 + beta_2 * y_2

class SCConv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 num_gn=1,
                 threshold=0.5, 
                 alpha=0.5,
                 squeeze_ratio=2):
        super().__init__()
        self.sru = SRU(num_gn, in_channels, threshold=threshold)
        self.cru = CRU(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio)

    def forward(self, x):
        x = self.sru(x)
        x = self.cru(x)
        return x
