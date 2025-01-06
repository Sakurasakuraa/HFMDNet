import torch
import torch.nn as nn
from modules.ASPP import ASPP
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


class Fushion(nn.Module):
    def __init__(self) :
        super(Fushion, self).__init__()
        self.aspp = ASPP()
        self.cr_branch_conv1 = nn.Sequential(
            BasicConv2d(256, 512, 3, padding=1),
            BasicConv2d(512, 256, 3, padding=1)
        )
        self.cr_max_pool_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1))
        self.cr_avg_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1))

        self.rd_branch_conv1 = nn.Sequential(
            BasicConv2d(256, 512, 3, padding=1),
            BasicConv2d(512, 256, 3, padding=1)
        )
        self.rd_max_pool_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1))
        self.rd_avg_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1))

        self.cd_branch_conv1 = nn.Sequential(
            BasicConv2d(256, 512, 3, padding=1),
            BasicConv2d(512, 256, 3, padding=1)
        )
        self.cd_max_pool_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1))
        self.cd_avg_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1))

        self.conv = BasicConv2d(1536,512,3,padding=1)

        self.sigmoid = nn.Sigmoid()



    def forward(self, input_rgb, input_depth,input_rgbd):


        r5_aspp = self.aspp(input_rgb)
        c5_aspp = self.aspp(input_rgbd)
        d5_aspp = self.aspp(input_depth)
        cr_fushion_0 = c5_aspp.mul(r5_aspp)
        cd_fushion_0 = c5_aspp.mul(d5_aspp)
        rd_fushion_0 = r5_aspp.mul(d5_aspp)

        cr_fushion_1 = self.cr_branch_conv1(cr_fushion_0)
        cr_fushion_max = self.cr_max_pool_branch(cr_fushion_1)
        cr_fushion_avg = self.cr_avg_pool_branch(cr_fushion_1)
        cr_weight = self.sigmoid(cr_fushion_max+cr_fushion_avg)


        rd_fushion_1 = self.rd_branch_conv1(rd_fushion_0)
        rd_fushion_max = self.rd_max_pool_branch(rd_fushion_1)
        rd_fushion_avg = self.rd_avg_pool_branch(rd_fushion_1)
        rd_weight = self.sigmoid(rd_fushion_max+rd_fushion_avg)

        cd_fushion_1 = self.cd_branch_conv1(cd_fushion_0)
        cd_fushion_max = self.cd_max_pool_branch(cd_fushion_1)
        cd_fushion_avg = self.cd_avg_pool_branch(cd_fushion_1)
        cd_weight = self.sigmoid(cd_fushion_max+cd_fushion_avg)

        rd_cr = rd_fushion_1 * cr_weight
        feature1 = rd_cr + rd_fushion_1

        cd_cr = cd_fushion_1 * (1-cr_weight)
        feature2 = cd_cr + cd_fushion_1

        cr_rd = cr_fushion_1 * rd_weight
        feature3 = cr_rd + cr_fushion_1

        cd_rd = cd_fushion_1 * (1-rd_weight)
        feature4 = cd_rd + cd_fushion_1

        cr_cd = cr_fushion_1 * cd_weight
        feature5 = cr_cd + cr_fushion_1

        rd_cd = rd_fushion_1 * (1-cd_weight)
        feature6 = rd_cd + rd_fushion_1

        fushion = self.conv(torch.cat((feature1,feature2,feature3,feature4,feature5,feature6), 1))



        return fushion