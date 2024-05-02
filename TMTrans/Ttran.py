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
from dedanet_v2 import MixVisionTransformer
from blocks import ResidualBlock,simam
from OTrans import OTransformer
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

class SkipConv(nn.Module):
    def __init__(self,in_c):
        super(SkipConv, self).__init__()
        self.conv1x1 = nn.Conv2d(in_c,in_c,kernel_size=1,padding=0)
        self.conv3x3 = nn.Conv2d(in_c,in_c,kernel_size=3,padding=1)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,skip):
        x = self.conv3x3(skip)
        x1 = self.conv1x1(skip)
        skip = self.relu(x+x1)
        return skip

class EncoderBlock(nn.Module):
    def __init__(self,in_c,k,p,f,in_g,embed_dims=[64, 128, 256,512]):
        super(EncoderBlock, self).__init__()
        self.r1 = ResidualBlock(in_c=in_c,out_c=in_c,k=k,p=p)
        self.r2 = ResidualBlock(in_c=in_c, out_c=in_c, k=k, p=p)
        self.t1 = MixVisionTransformer(in_chans=in_c,floor=f,depths=[1],in_g=in_g)
        self.conv1x1 = nn.Conv2d(in_c,embed_dims[f],kernel_size=1,padding=0)
        self.conv1x1_ = nn.Conv2d(embed_dims[f], embed_dims[f], kernel_size=1, padding=0)
        self.simam = simam()
        self.bn = nn.BatchNorm2d(embed_dims[f])
        self.relu = nn.ReLU(inplace=True)
        self.maxp = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, xg):
        x1 = self.r1(x)
        x2 = self.r2(x1)
        # print('Encoder r1  x ',x.shape)
        xs = self.maxp(self.relu(self.bn(self.conv1x1(x1))))
        # print('Encoder x1  x ', x1.shape)
        x2 = self.simam(self.t1(x2, xg))
        # print(x.shape, '  --t1--  ', x1.shape)
        out = x2 + xs
        #out = self.simam(self.conv1x1_(out))   x2 simam 取消
        out = self.conv1x1_(out)
        skip = out
        #print('..',out.shape,skip.shape)
        return out, skip

        
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c,k,p, name=None):
        super(DecoderBlock, self).__init__()
        self.r1 = ResidualBlock(in_c*3, out_c,k,p)
        self.r2 = ResidualBlock(out_c, out_c,k,p)
        self.skconv = SkipConv(in_c)
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_c*2,in_c*2,1)

    def forward(self, inputs, skip,gt):
        skip = self.skconv(self.bn(self.skconv(self.skconv(skip))))
        x = torch.cat([inputs, skip], axis=1)
        x = self.conv1x1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        gt = nn.functional.interpolate(gt, scale_factor=2, mode='bilinear',align_corners=True)
        x = torch.cat([x,gt],dim=1)
        x = self.r1(x)
        x = self.r2(x)
        return x

class DecoderBlock1(nn.Module):
    def __init__(self, in_c, out_c,k,p, name=None):
        super(DecoderBlock1, self).__init__()
        self.r1 = ResidualBlock(in_c + in_c, out_c,k,p)
        self.r2 = ResidualBlock(out_c, out_c,k,p)
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()
        self.skconv =SkipConv(in_c)


    def forward(self, inputs, skip):
        skip = self.skconv(self.bn(self.skconv(self.skconv(skip))))
        x = torch.cat([inputs, skip], axis=1)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        x = self.r1(x)
        x = self.r2(x)
        return x

class GLCMEncoder(nn.Module):
    def __init__(self,in_c,out_c):
        super(GLCMEncoder, self).__init__()
        self.r1 = ResidualBlock(in_c,out_c,3,1)
        self.r2 = ResidualBlock(out_c, out_c, 3, 1)
        self.maxpool=nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        x=self.r1(x)
        x=self.r2(x)
        x=self.maxpool(x)
        return x

class Maxpfusion(nn.Module):
    def __init__(self,in_c,out_c):
        super(Maxpfusion, self).__init__()
        self.conv1x1=nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=1)
        self.se=SELayer(out_c)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,m1,m2):
        m=torch.cat([m1,m2],dim=1)
        x=self.relu(self.se(self.conv1x1(m)))
        return x

class GLCMTransF(nn.Module):
    def __init__(self,in_c,out_c):
        super(GLCMTransF, self).__init__()
        self.conv3x3=nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=3,padding=1)
        self.BN=nn.BatchNorm2d(out_c)
        self.ReLU=nn.ReLU(inplace=True)
        self.Sigmoid=nn.Sigmoid()
    def forward(self,g,t):
        g=self.Sigmoid(g)
        g=g> 0.5
        g_t = g*t
        g_t = self.conv3x3(g_t)
        t1=self.ReLU(self.BN(self.conv3x3(t)))
        out=g_t*t1
        return out

class TWLNet(nn.Module):
    def __init__(self):
        super(TWLNet, self).__init__()
        self.e1 = OTransformer(in_chans=3,floor=0,depths=[3])
        self.e2 = EncoderBlock(64, 3, 1, 1,64)
        self.e3 = EncoderBlock(128, 3, 1, 2,128)
        self.e4 = EncoderBlock(256, 3, 1,3,256)

        self.eg1 = GLCMEncoder(in_c=1,out_c=64)
        self.eg2 = GLCMEncoder(in_c=64, out_c=128)
        self.eg3 = GLCMEncoder(in_c=128, out_c=256)
        self.eg4 = GLCMEncoder(in_c=256, out_c=512)

        self.GLCMTransF1=GLCMTransF(in_c=64,out_c=64)
        self.GLCMTransF2=GLCMTransF(in_c=128,out_c=128)
        self.GLCMTransF3=GLCMTransF(in_c=256,out_c=256)
        self.max_pool = nn.AdaptiveMaxPool2d(32)
        self.max_pool1 = nn.AdaptiveMaxPool2d(16)
        self.maxpf=Maxpfusion(128+256,256)
        self.maxpf1 = Maxpfusion(256 + 512, 512)

        self.d1 = DecoderBlock1(512,256,3,1)
        self.d2 = DecoderBlock(256,128,3,1)
        self.d3 = DecoderBlock(128,64,3,1)
        self.d4 = DecoderBlock(64,16,3,1)

        self.conv1x1 = nn.Conv2d(1,1,kernel_size=1, padding=0)
        self.output = nn.Conv2d(16,1,kernel_size=1, padding=0)
        self.linear2 = nn.Linear(64, 1024)
        self.linear3 = nn.Linear(16, 1024)
        
        
    def forward(self, input):
        x,GLCM =input[0],input[1]
        e1,s1= self.e1(x)
        g1 = self.eg1(GLCM)

        e2,s2= self.e2(e1,g1)
        #print('  .  ', e1.shape, s1.shape, g1.shape)
        #print('  .1  ', e2.shape, s2.shape)
        maxp_lay2= self.max_pool(e2)
        #print('  .2  ',maxp_lay2.shape)
        g2 = self.eg2(g1)

        e3,s3 = self.e3(e2,g2)
        #print('  .3  ', maxp_lay2.shape, s3.shape,e3)
        me3=self.maxpf(maxp_lay2,e3)
        #print('   . 4 ',me3.shape)
        maxp_lay3 = self.max_pool1(me3)
        g3 = self.eg3(g2)

        e4,s4 = self.e4(e3,g3)
        me4=self.maxpf1(maxp_lay3,e4)


        gt1=self.GLCMTransF1(g1,e1)
        gt2 = self.GLCMTransF2(g2, e2)
        gt3 = self.GLCMTransF3(g3, e3)


        d4 = self.d1(me4,s4)
        d3 = self.d2(d4,s3,gt3)
        d2 = self.d3(d3,s2,gt2)
        d1 = self.d4(d2,s1,gt1)
        out = self.output(d1)
        return out
        
if __name__=='__main__':
    img = torch.randn(2, 3, 224, 224)
    glcm= torch.randn(2, 1, 224, 224)
    t = TWLNet()

    preds = t([img,glcm])  # (1,1000)
    print(preds.shape)
    from torchsummary import summary
    from thop import profile
    from thop import  clever_format
    flops,params = profile(t,inputs=([img,glcm],))
    print('%.3f  |  %.3f  ' %(params/(1000**2),flops/(1000**3)))
    #54.511   118737
    #print(preds.shape)

