import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class FE_low(nn.Module):
    def __init__(self, in_channel):
        super(FE_low, self).__init__()
        self.rgb_branch = BasicConv2d(in_channel, in_channel, 3, padding=1, dilation=1)
        self.d_branch = BasicConv2d(in_channel, in_channel, 3, padding=1, dilation=1)
        self.r_d_sa = SpatialAttention()
        self.d_r_sa = SpatialAttention()

        self.r_aap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 3, padding=1)
            )
        self.d_aap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 3, padding=1)
            )

        self.softmax = nn.Softmax(dim=1)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(std=0.01)
        #         m.bias.data.fill_(0)
        self.conv_rgb = BasicConv2d(2 * in_channel, in_channel, 3, padding=1, dilation=1)
        self.conv_d = BasicConv2d(2 * in_channel, in_channel, 3, padding=1, dilation=1)

    def forward(self, x_rgb, x_d):
        x_rgb = self.rgb_branch(x_rgb)
        x_d = self.d_branch(x_d)
        # rgb增强depth
        # 支路1
        x1_rgb_sa = x_rgb.mul(self.r_d_sa(x_rgb))
        # 支路2
        x_rgb_aap = self.r_aap(x_rgb)
        x2_r_map = self.softmax(x_rgb_aap)
        x2_rgb = torch.cat((x_rgb.mul(x2_r_map), x_rgb), 1)

        x_rgb_s = x1_rgb_sa.mul(self.conv_rgb(x2_rgb)) + x1_rgb_sa + self.conv_rgb(x2_rgb)
        x1_d_FE = x_d.mul(x_rgb_s)
        # 支路1
        x1_d_sa = x1_d_FE.mul(self.d_r_sa(x1_d_FE))

        # 支路2
        x_d_aap = self.d_aap(x_rgb)
        x2_d_map = self.softmax(x_d_aap)
        x2_d = torch.cat((x1_d_FE.mul(x2_d_map), x1_d_FE), 1)

        x_d_s = x1_d_sa.mul(self.conv_d(x2_d)) + x1_d_sa + self.conv_d(x2_d)

        x1_rgb_FE = x_rgb.mul(x_d_s)

        return x1_rgb_FE, x1_d_FE