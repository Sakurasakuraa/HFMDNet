import torch
import torch.nn as nn
from modules.FE_high import FE_high

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

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.CA = ChannelAttention(1024)


        self.SA3_2 = SpatialAttention()
        self.SA3_1 = SpatialAttention()
        self.SA2_1 = SpatialAttention()

        #self.feature_conv = nn.Sequential(BasicConv2d(1024, 1024, 3, padding=1),)
        self.feature_conv = FE_high(1024, 1024)

        self.sal_up_3 = nn.Sequential(
            self.upsample2,
            BasicConv2d(512, 512, 3, padding=1),

        )
        self.sal_up_2 = nn.Sequential(
            self.upsample4,
            BasicConv2d(512, 256, 3, padding=1),

        )
        self.sal_up_1 = nn.Sequential(
            self.upsample4,
            BasicConv2d(512, 256, 3, padding=1),
            self.upsample2,
            BasicConv2d(256, 128, 3, padding=1),
        )



        self.sal_pre = nn.Sequential(
            BasicConv2d(512, 256, 3, padding=1),
            self.upsample4,
            BasicConv2d(256, 64, 3, padding=1),
            self.upsample4,
            BasicConv2d(64, 32, 3, padding=1),
            self.upsample2,
            nn.Conv2d(32, 1, 3, padding=1),
        )

        self.conv1024_512 = BasicConv2d(1024, 512, 3, padding=1)

        self.d3 = nn.Sequential(
            BasicConv2d(1024, 512, 3, padding=1),
        )

        self.s3_pre = nn.Sequential(
            BasicConv2d(512, 128, 3, padding=1),
            self.upsample4,
            BasicConv2d(128, 64, 3, padding=1),
            self.upsample4,
            nn.Conv2d(64, 1, 3, padding=1),
        )

        self.s3_up_2 = nn.Sequential(
            self.upsample2,
            BasicConv2d(512, 256, 3, padding=1),

        )

        self.s3_up_1 = nn.Sequential(
            self.upsample4,
            BasicConv2d(512, 128, 3, padding=1),

        )

        self.s2_1 = nn.Sequential(
            self.upsample2,
            BasicConv2d(256, 128, 3, padding=1),
        )

        self.l2_a_conv = BasicConv2d(512, 256, 3, padding=1)

        self.d2 = nn.Sequential(
            BasicConv2d(512, 256, 3, padding=1),
        )

        self.s2_pre = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1),
            self.upsample4,
            BasicConv2d(128, 64, 3, padding=1),
            self.upsample2,
            nn.Conv2d(64, 1, 3, padding=1),
        )

        self.l1_a_conv = BasicConv2d(256, 128, 3, padding=1)
        self.l1_b_conv = BasicConv2d(256, 128, 3, padding=1)

        self.d1 = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1),
            BasicConv2d(128, 64, 3, padding=1),
        )
        self.s1_pre = nn.Sequential(
            BasicConv2d(96, 32, 3, padding=1),
            self.upsample4,
            nn.Conv2d(32, 1, 3, padding=1)
        )



    def forward(self, feature_list,edge):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        feature1 = feature_list[0]  #128*96*96
        feature2 = feature_list[1]  #256*48*48
        feature3 = feature_list[2]  #512*24*24
        feature4 = feature_list[3]  #1024*12*12
        feature5 = feature_list[4]  #1024*12*12

        feature5_conv = self.feature_conv(feature5, feature4) #1024*12*12
        #feature5_conv = self.sal_conv(torch.cat((feature5_conv * feature4, feature5_conv, feature5_conv + feature4), dim=1)) #1024*12*12
        high_sal = self.CA(feature5_conv) * feature5_conv #1024*12*12
        high_sal = self.conv1024_512(high_sal)
        s4 = self.sal_pre(high_sal)  #1*384*384

        high_sal_3 = self.sal_up_3(high_sal) #512*24*24
        high_sal_2 = self.sal_up_2(high_sal) #256*48*48
        high_sal_1 = self.sal_up_1(high_sal) #128*96*96

        l3 = torch.cat((high_sal_3 + feature3, high_sal_3), dim=1)
        s3_sal = self.d3(l3)  #512*24*24
        s3 = self.s3_pre(s3_sal) #1*384*384
        s3_sal_2 = self.s3_up_2(s3_sal) #256*48*48
        s3_sal_SA_2 = self.SA3_2(s3_sal_2) * s3_sal_2
        s3_sal_1 = self.s3_up_1(s3_sal) #128*96*96
        s3_sal_SA_1 = self.SA3_1(s3_sal_1) * s3_sal_1

        l2_a = torch.cat((high_sal_2 + feature2, high_sal_2), dim=1)
        l2_a_conv = self.l2_a_conv(l2_a)  #256*48*48
        l2 = torch.cat((l2_a_conv, s3_sal_SA_2), dim=1)
        s2_sal = self.d2(l2) #256*48*48
        s2 = self.s2_pre(s2_sal) #1*384*384
        s2_sal_1 = self.s2_1(s2_sal) #128*96*96
        s2_sal_SA_1 = self.SA2_1(s2_sal_1) * s2_sal_1

        l1_a = torch.cat((high_sal_1 + feature1, high_sal_1), dim=1)
        l1_a_conv = self.l1_a_conv(l1_a)  #128*96*96
        l1_b = torch.cat((l1_a_conv, s2_sal_SA_1), dim=1)
        l1_b_conv = self.l1_b_conv(l1_b)  #128*96*96
        l1 = torch.cat((l1_b_conv, s3_sal_SA_1), dim=1)
        s1_sal = self.d1(l1) #64*96*96

        s1_edge = self.relu(torch.cat((s1_sal, edge), dim=1))
        s1 = self.s1_pre(s1_edge) #1*384*384

        return  s1, s2, s3, s4