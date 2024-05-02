import warnings
import torch.nn.functional as F
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_
import math
from timm.models.layers import DropPath
from torch.nn import Module
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn
import torch
import numpy as np
from OTrans import OTransformer
from blocks import ResidualBlock,simam,FFTBlock,BLEBlock
from FFTConv import SpectralTransform
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

class GobalFFTConv(nn.Module):
    def __init__(self,channel,k,p):
        super(GobalFFTConv, self).__init__()
        self.FFTconv = SpectralTransform(channel[0], channel[1])
        self.BKConv = ResidualBlock(channel[1], channel[1], k, p)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        ft = self.FFTconv(x)
        bk = self.BKConv(ft)
        return bk

class LocalBConv(nn.Module):
    def __init__(self,channel):
        super(LocalBConv, self).__init__()
        self.r1 = ResidualBlock(channel[0], channel[1], 3,1)
        self.r2 = ResidualBlock(channel[1], channel[1], 3,1)

    def forward(self,x):
        x1 = self.r1(x)
        out = self.r2(x1)
        return out

class Decoder(nn.Module):
    def __init__(self,channel):
        super(Decoder, self).__init__()
        self.r1=ResidualBlock(channel[0],channel[1],3,1)
        self.r2=ResidualBlock(channel[1],channel[1],3,1)
        self.CatConv = nn.Conv2d(channel[0]*2,channel[0],kernel_size=3,padding=1)
        self.skconv = SkipConv(64)
    def forward(self,x,skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        skip = nn.functional.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=True)
        skip = self.skconv(skip)
        cx = torch.cat([x,skip],axis=1)
        cx = self.CatConv(cx)
        out = self.r2(self.r1(cx))
        return out

class Decoder1(nn.Module):
    def __init__(self,channel):
        super(Decoder1, self).__init__()
        self.r1=ResidualBlock(channel[0],channel[1],3,1)
        self.r2=ResidualBlock(channel[1],channel[1],3,1)
        self.CatConv = nn.Conv2d(channel[0]*3,channel[0],kernel_size=3,padding=1)
        self.simam = simam()
        self.skconv = SkipConv(channel[0])

    def forward(self,x,skip,m):
        #print(m.shape,'   m shape', x.shape, '  x shape ', skip.shape)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        skip = nn.functional.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=True)
        #print(x.shape,skip.shape)
        cx = torch.cat([x,skip,self.simam(m)],axis=1)
        cx = self.CatConv(cx)
        #print('cx shape  ', cx.shape)
        out = self.r2(self.r1(cx))
        return out

class SkipConv(nn.Module):
    def __init__(self,in_c):
        super(SkipConv, self).__init__()
        self.conv1x1 = nn.Conv2d(in_c,in_c,kernel_size=1,padding=0)
        self.conv3x3 = nn.Conv2d(in_c,in_c,kernel_size=3,padding=1)
        self.relu=nn.ReLU(inplace=True)
        self.simam = simam()

    def forward(self,skip):
        x = self.conv3x3(skip)
        x1 = self.conv1x1(skip)
        skip = self.simam(self.relu(x+x1))
        return skip

class Encoder(nn.Module):
    def __init__(self,channel,f,cf):
        super(Encoder, self).__init__()
        self.r1=ResidualBlock(channel[0],channel[1],3,1)
        self.r2 = ResidualBlock(channel[1], channel[1], 3, 1)
        self.FFT = SpectralTransform(channel[1],channel[1])
        self.down = nn.Conv2d(channel[1],channel[1],kernel_size=3,padding=1,stride=1)
        self.avg = nn.AvgPool2d(kernel_size=2)
        self.conv = nn.Conv2d(channel[1]*2,channel[1],kernel_size=1,padding=0)
        self.simam = simam()
        self.bn = nn.BatchNorm2d(channel[1])
        self.DTrans = OTransformer(channel[1],floor=f,cf=cf,depths=[1])

    def forward(self,x):
        x1 = self.r2(self.r1(x))
        mpx = x1
        x1 = self.avg(x1)
        fft = self.simam(self.down(self.FFT(x1)))+x1
        out = self.bn(self.conv(torch.cat([x1,fft],axis=1)))
        skip = self.DTrans(x1,fft)
        return out,skip,mpx

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(256,256*2)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(256*2,256)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        fx = self.drop(self.act(self.fc(x)))
        fx1 = self.drop(self.fc1(fx))
        return fx1

class MultiScaleFusionMLP(nn.Module):
    def __init__(self):
        super(MultiScaleFusionMLP,self).__init__()
        self.proj_inm1 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.proj_inm2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.proj_inm3 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.proj_outm1 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.proj_outm2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.proj_outm3 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.MLP = MLP()

    def forward(self, m1, m2, m3, B):
        viewm1 = self.proj_inm1(m1).view(B, 256, -1).permute(0, 2, 1)
        viewm2 = self.proj_inm2(m2).view(B, 256, -1).permute(0, 2, 1)
        viewm3 = self.proj_inm3(m3).view(B, 256, -1).permute(0, 2, 1)
        catview = torch.cat([viewm1, viewm2, viewm3], dim=1)
        catview = self.MLP(catview)
        outm1 = catview[:, :16384, :]
        outm2 = catview[:, 16384:16384+4096, :]
        outm3 = catview[:, 16384+4096:, :]
        outm1 = self.proj_outm1(outm1.permute(0, 2, 1).view(B, 256, 128, 128))
        outm2 = self.proj_outm2(outm2.permute(0, 2, 1).view(B, 256, 64, 64))
        outm3 = self.proj_outm3(outm3.permute(0, 2, 1).view(B, 256, 32, 32))

        return outm1,outm2,outm3

class FFTDnet(nn.Module):
    def __init__(self):
        super(FFTDnet, self).__init__()
        self.InitConv = nn.Conv2d(3,3,kernel_size=3,padding=1)
        self.e1 = Encoder([3,64],0,1)
        self.e2 = Encoder([64, 128],1,2)
        self.e3 = Encoder([128, 256],2,3)
        self.e4 = Encoder([256, 512],3,4)

        self.msf = MultiScaleFusionMLP()

        self.d1 = Decoder1([512,256])
        self.d2 = Decoder1([256, 128])
        self.d3 = Decoder1([128, 64])
        self.d4 = Decoder([64, 3])

        self.out = nn.Conv2d(3,1,kernel_size=1,padding=0)
        self.tout = nn.Conv2d(1, 1, kernel_size=1, padding=0)
        self.maxp = nn.MaxPool2d(kernel_size=2)
        self.simam = simam()
        self.bn = nn.BatchNorm2d(512)

    def forward(self, input):
        B = input.shape[0]
        input = self.InitConv(input)
        #bleout = self.BLE(input)
        e1, s1, m = self.e1(input)
        e2, s2, m1 = self.e2(e1)
        e3, s3, m2 = self.e3(e2)
        e4, s4, m3 = self.e4(e3)
        #print(m1.shape, '   ', m2.shape, '   ', m3.shape )
        m1, m2, m3 = self.msf(m1, m2, m3, B)

        d1 = self.d1(e4, s4, m3)
        d2 = self.d2(d1, s3, m2)
        d3 = self.d3(d2, s2, m1)
        d4 = self.d4(d3, s1)
        out = self.out(d4)

        return out
        
if __name__=='__main__':
    img = torch.randn(2, 3, 256, 256)
    t = FFTDnet()
    preds = t(img)  # (1,1000)
    print(preds.shape)
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(t, input_res=(3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)#103.04
    print('      - Params: ' + params)#51.05

