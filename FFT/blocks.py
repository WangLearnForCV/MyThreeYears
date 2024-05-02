import torch
import torch.nn as nn
from  FFTConv import FFTConv2d
import numpy as np
""" Squeeze and Excitation block """
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class simam(nn.Module):
    def __init__(self,e_lambda=1e-4):
    	super(simam, self).__init__()
    	self.activaton = nn.Sigmoid()
    	self.e_lambda=e_lambda
    	
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
        
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)    

""" 3x3->3x3 Residual block """
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c,k,p):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k, padding=p)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=k, padding=p)
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



""" Mixpool block: Merging the image features and the mask """
class MixPool(nn.Module):
    def __init__(self, in_c, out_c):
        super(MixPool, self).__init__()

        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )
        
        

    def forward(self, x, m):
        fmask = (self.fmask(x) > 0.5).type(torch.cuda.FloatTensor)
        m = nn.MaxPool2d((m.shape[2]//x.shape[2], m.shape[3]//x.shape[3]))(m)
        x1 = x * torch.logical_or(fmask, m).type(torch.cuda.FloatTensor)
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], axis=1)
        return x

class MixPool_with_And(nn.Module):
    def __init__(self, in_c, out_c):
        super(MixPool_with_And, self).__init__()

        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//4),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )
        self.conv1x1=nn.Conv2d(1,1,kernel_size=1)

    def forward(self, x, m):
        fmask = (self.fmask(x) > 0.5).type(torch.cuda.FloatTensor)
        m = nn.MaxPool2d((m.shape[2]//x.shape[2], m.shape[3]//x.shape[3]))(m)
        m = (self.conv1x1(m) > 0.5).type(torch.cuda.FloatTensor)
        x1 = x * torch.logical_or(fmask, m).type(torch.cuda.FloatTensor)
        x3 = x * torch.logical_and(fmask,m).type(torch.cuda.FloatTensor)
        x1 = self.conv1(x1)
        x3 = self.conv1(x3)
        x2 = self.conv2(x)

        x = torch.cat([x1, x2, x3], axis=1)
        return x


class FFTBlock(nn.Module):
    def __init__(self,channels):
        super(FFTBlock, self).__init__()
        self.fft12 = FFTConv2d(channels[0],channels[1], 3, 1, 1, True)
        self.fft23 = FFTConv2d(channels[1], channels[1], 3, 1, 1, True)
        self.fft13 = FFTConv2d(channels[0], channels[1], 1, 1, 0, True)
        self.BN = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.se = simam()
    def forward(self,input):
        x1 = self.fft12(input)
        x1 = self.relu(self.BN(x1))
        x2 = self.se(self.fft23(x1))
        x3 = self.fft13(input)

        out = x2+x3
        return out

class BLEBlock(nn.Module):
    def __init__(self, channel):
        super(BLEBlock, self).__init__()
        self.conv7x7 = nn.Conv2d(channel[0], channel[1], kernel_size=7, padding=3)
        self.conv1x7 = nn.Conv2d(channel[1], channel[1], kernel_size=(1, 3), padding=(0, 1))
        self.conv7x1 = nn.Conv2d(channel[0], channel[1], kernel_size=(3, 1), padding=(1, 0))
        self.conv3x3 = nn.Conv2d(channel[0], channel[1], kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(channel[1], channel[1], kernel_size=1, padding=0)
        #self.outconv = nn.Conv2d(channel[1]*2,channel[1],kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(channel[1])
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        b7 = self.relu(self.bn(self.conv7x7(input)))
        b71 = self.conv7x1(input)
        b171 = self.conv1x1(self.bn(self.conv1x7(b71)))
        bqk = self.sigmoid(self.conv1x1(b7*b171))
        b3 = self.bn((self.conv3x3(input)))
        out = bqk*b3

        return out
