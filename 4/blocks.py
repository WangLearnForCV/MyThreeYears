 import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
""" Squeeze and Excitation block """


class simam(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se = simam()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.se(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4


class DWConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DWConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class TSC_ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(TSC_ResBlock, self).__init__()
        self.Conv3x3 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn3x3 = nn.InstanceNorm2d(ch_out)
        self.ReLU = nn.ReLU(inplace=True)

        self.DConv3x3 = nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=3, padding=2, dilation=2)
        self.Channel_bn = nn.InstanceNorm2d(ch_out // 2)
        self.Channel_bn_ = nn.InstanceNorm2d(ch_out // 4)

        self.Conv1x1 = nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1)
        self.Conv3x3_ = nn.Conv2d(ch_out // 2, ch_out // 2, kernel_size=3, padding=1)

        self.Conv1x1_ = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.simam = simam()
        self.Conv3x3o = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.ReLU(self.bn3x3(self.Conv3x3(input)))
        x1, x1_ = x.chunk(2, dim=1)
        x2, x3 = x1_.chunk(2, dim=1)
        x1 = self.Channel_bn(self.Conv3x3_(x1))
        x2 = self.Channel_bn_(self.DConv3x3(x2))
        x3 = self.Channel_bn_(self.Conv1x1(x3))
        xo = self.ReLU(self.simam(self.Conv3x3o(torch.cat([x1, x2, x3], dim=1))))
        xb = self.ReLU(self.bn3x3(self.Conv1x1_(input)))
        out = xb + xo

        return out


class TCIEncoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(TCIEncoder, self).__init__()
        self.r1 = TSC_ResBlock(ch_in, ch_out)
        self.r2 = TSC_ResBlock(ch_out, ch_out)

    def forward(self, x):
        x1 = self.r1(x)
        x2 = self.r2(x1)
        return x2, x2


class ResEncoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResEncoder, self).__init__()
        self.r1 = ResidualBlock(ch_in, ch_out)
        self.r2 = ResidualBlock(ch_out, ch_out)

    def forward(self, x):
        x1 = self.r1(x)
        x2 = self.r2(x1)
        return x2, x2


class TCIDecoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(TCIDecoder, self).__init__()
        self.r1 = TSC_ResBlock(ch_in, ch_in)
        self.r2 = TSC_ResBlock(ch_in, ch_out)

    def forward(self, x, s, isup):
        r1 = self.r1(x)
        if isup:
            r1 = nn.functional.interpolate(r1, scale_factor=2, mode='bilinear', align_corners=True)
            r1 = r1 + s
        r2 = self.r2(r1)
        return r2


class ResDecoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResDecoder, self).__init__()
        self.r1 = ResidualBlock(ch_in, ch_out)
        self.r2 = ResidualBlock(ch_out, ch_out)

    def forward(self, x, s, isup):
        if isup:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = x + s
        x1 = self.r1(x)
        x2 = self.r2(x1)
        return x2


class MLP(nn.Module):
    def __init__(self, ch_in, sch):
        super(MLP, self).__init__()
        self.proj_inm1 = nn.Conv2d(ch_in, ch_in, kernel_size=1)
        self.proj_inm2 = nn.Conv2d(sch, ch_in, kernel_size=1)
        self.proj_out = nn.Conv2d(ch_in, ch_in, kernel_size=1)
        self.fc = nn.Linear(ch_in, ch_in * 2)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(ch_in * 2, ch_in)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, x2):
        b, c, h, w = x.size()
        x = self.proj_inm1(x).view(b, c, -1).permute(0, 2, 1)
        x2 = self.proj_inm2(x2).view(b, c, -1).permute(0, 2, 1)
        mlp = torch.cat([x, x2], dim=1)
        mlp = self.drop(self.act(self.fc(mlp)))
        mlp = self.drop(self.fc1(mlp))
        out, out1 = mlp.chunk(2, dim=1)
        out = self.proj_out(out.permute(0, 2, 1).view(b, c, h, w))
        return out


class CIBlock(nn.Module):
    # ch[32,64,128]
    def __init__(self, ch):
        super(CIBlock, self).__init__()
        self.Conv3x3 = nn.Conv2d(ch[0], ch[0], kernel_size=3, padding=1)
        self.Channel_bn = nn.InstanceNorm2d(ch[0])
        self.c3x3 = nn.Conv2d(ch[0] + ch[1], ch[0] * 2, kernel_size=3, padding=1)

        self.Conv3x3_ = nn.Conv2d(ch[1], ch[1], kernel_size=3, padding=1)
        self.Channel_bn_ = nn.InstanceNorm2d(ch[1])
        self.c3x3_ = nn.Conv2d(ch[0] + ch[2], ch[1] * 2, kernel_size=3, padding=1)

        self.Conv3x3__ = nn.Conv2d(ch[2], ch[2], kernel_size=3, padding=1)
        self.Channel_bn__ = nn.InstanceNorm2d(ch[2])
        self.c3x3__ = nn.Conv2d(ch[1] + ch[2], ch[2] * 2, kernel_size=3, padding=1)

        self.maxp = nn.MaxPool2d(2)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, s1, s2, s3):
        x1, x1_ = s1.chunk(2, dim=1)
        x1 = self.Channel_bn(self.Conv3x3(x1))
        x1_ = self.Channel_bn(self.Conv3x3(self.maxp(x1_)))

        x2, x2_ = s2.chunk(2, dim=1)
        x2 = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = self.Channel_bn_(self.Conv3x3_(x2))
        x2_ = self.Channel_bn_(self.Conv3x3_(self.maxp(x2_)))

        x3, x3_ = s3.chunk(2, dim=1)
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = self.Channel_bn__(self.Conv3x3__(x3))
        x3_ = self.Channel_bn__(self.Conv3x3__(x3_))

        s1_ = torch.cat([x1, x2], dim=1)
        s1 = s1 + self.ReLU(self.c3x3(s1_))

        s2_ = torch.cat([x1_, x3], dim=1)
        s2 = s2 + self.ReLU(self.c3x3_(s2_))

        s3_ = torch.cat([x2_, x3_], dim=1)
        s3 = s3 + self.ReLU(self.c3x3__(s3_))

        return s1, s2, s3


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, k=0.3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = self.gauss(3, k)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels).to('cuda')
        return x

    def gauss(self, kernel_size, sigma):
        kernel1 = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2 = cv2.getGaussianKernel(kernel_size, sigma)
        kernel3 = np.multiply(kernel1, np.transpose(kernel2))
        return kernel3

class DFGBlock(nn.Module):
    def __init__(self):
        super(DFGBlock, self).__init__()
        self.GSConv = GaussianBlurConv(3, 0.3)
        self.GSConv1 = GaussianBlurConv(3, 0.4)
        self.GSConv2 = GaussianBlurConv(3, 0.5)
        self.GSConv3 = GaussianBlurConv(3, 0.6)
        self.GSConv4 = GaussianBlurConv(3, 0.7)
        self.GSConv5 = GaussianBlurConv(3, 0.8)

        self.conv0 = nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.deconv0 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.deconv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self,x):
        dog1 = self.GSConv(x)
        dog2 = self.GSConv1(x)
        dog3 = self.GSConv2(x)
        dog4 = self.GSConv3(x)
        dog5 = self.GSConv4(x)
        dog6 = self.GSConv5(x)
        g1 = torch.sub(dog2, dog1)
        g2 = torch.sub(dog4, dog3)
        g3 = torch.sub(dog6, dog5)
        dog = torch.cat([g1, g2, g3], dim=1)

        e1 = self.conv0(dog)
        e2 = self.conv1(e1)
        e3 = self.conv2(e2)

        d3 = self.deconv0(e3)
        d3 = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.deconv1(d3)
        d2 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.deconv2(d2)

        return d1
