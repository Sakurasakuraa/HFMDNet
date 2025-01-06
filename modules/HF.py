import torch
import torch.nn as nn

from torch import Tensor

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

class Fushion(nn.Module):
    def __init__(self, in_channel, out_channel) :
        super(Fushion, self).__init__()

        self.r_ca = ChannelAttention(in_channel)
        self.c_ca = ChannelAttention(in_channel)
        self.d_ca = ChannelAttention(in_channel)
        self.conv1 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.conv3 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.cr_branch = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 3, padding=1))
        self.cd_branch = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 3, padding=1))
        self.max_pool_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(out_channel, 2 * out_channel, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(2 * out_channel, out_channel, 3, padding=1, dilation=1))
        self.avg_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, 2 * out_channel, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(2 * out_channel, out_channel, 3, padding=1, dilation=1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_rgb, input_depth,input_rgbd):


        r_ca = input_rgb.mul(self.r_ca(input_rgb))
        c_ca = input_rgbd.mul(self.c_ca(input_rgbd))
        d_ca = input_depth.mul(self.d_ca(input_depth))

        r = self.conv1(r_ca)
        c = self.conv2(c_ca)
        d = self.conv3(d_ca)

        cr_fushion_0 = c.mul(r)
        cd_fushion_0 = c.mul(d)
        rd_fushion = r.mul(d)

        cr_fushion_1 = self.cr_branch(cr_fushion_0)
        cd_fushion_1 = self.cd_branch(cd_fushion_0)

        rd_fushion_max = self.max_pool_branch(rd_fushion)
        rd_fushion_avg = self.avg_pool_branch(rd_fushion)
        rd_measure = self.sigmoid(rd_fushion_max+rd_fushion_avg)
        cr_fushion_2 = cr_fushion_1 * rd_measure
        cd_fushion_2 = cd_fushion_1 * (1-rd_measure)
        cr_fushion = cr_fushion_0 + cr_fushion_2
        cd_fushion = cd_fushion_0 + cd_fushion_2
        fushion = torch.cat((cr_fushion.mul(cd_fushion), cr_fushion+cd_fushion), 1)


        return fushion