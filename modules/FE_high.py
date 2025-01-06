import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FE_high(nn.Module):
    def __init__(self, in_channel, depth):
        super(FE_high, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.d_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.d_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.d_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=5, dilation=5)

        self.r_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.r_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.r_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=5, dilation=5)

        self.d_conv_output = nn.Conv2d(depth * 3, depth, 3, padding=1)
        self.r_conv_output = nn.Conv2d(depth * 3, depth, 3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.Bconv = BasicConv2d(depth, depth, 3, padding=1)

    def forward(self, rgb, d):

        image_features = self.mean(rgb)
        image_features = self.conv(image_features)
        r_weight = self.sigmoid(image_features)

        d_block1 = self.d_block1(d)
        d_block3 = self.d_block3(d)
        d_block6 = self.d_block6(d)

        r_block1 = self.r_block1(rgb)
        r_block3 = self.r_block3(rgb)
        r_block6 = self.r_block6(rgb)

        d_net = self.d_conv_output(torch.cat([d_block1, d_block3, d_block6], dim=1))
        d_weight = self.sigmoid(d_net)

        r_net = self.r_conv_output(torch.cat([r_block1, r_block3, r_block6], dim=1))

        rgb_1 = r_net * d_weight + r_net
        rgb_2 = rgb_1 * r_weight

        rgb_s = self.Bconv(rgb_2)

        return rgb_s
