import torch
import torch.nn as nn
from blocks import ResidualBlock, MixPool,SELayer,MixPool_with_And
import cv2
import numpy as np
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c,k,p,f_c, name=None):
        super(EncoderBlock, self).__init__()

        self.name = name
        self.r1 = ResidualBlock(in_c, out_c,k,p)
        self.r2 = ResidualBlock(out_c, out_c,k,p)
        self.p1 = MixPool_with_And(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        self.fr1 = ResidualBlock(f_c,out_c//4,k,p)
        self.conv1x1 = nn.Conv2d(out_c+out_c//4,out_c,kernel_size=1,padding=0)
        self.bn=nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, masks,ft):
        f = self.fr1(ft)
        x = self.r1(inputs)
        x = self.r2(x)
        x = torch.cat([x,f],axis=1)
        x = self.relu(self.bn(self.conv1x1(x)))
        p = self.p1(x, masks)
        o = self.pool(p)
        f = self.pool(f)
        return o, x,f

class EncoderBlock1(nn.Module):
    def __init__(self, in_c, out_c,k,p,f_c, name=None):
        super(EncoderBlock1, self).__init__()

        self.name = name
        self.r1 = ResidualBlock(in_c, out_c,k,p)
        self.r2 = ResidualBlock(out_c, out_c,k,p)
        self.p1 = MixPool_with_And(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        self.fr1 = ResidualBlock(f_c, out_c // 4, k, p)
        self.conv1x1 = nn.Conv2d(out_c + out_c // 4, out_c, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs,ft):
        f = self.fr1(ft)
        x = self.r1(inputs)
        x = self.r2(x)
        x = torch.cat([x, f], axis=1)
        x = self.relu(self.bn(self.conv1x1(x)))
        o = self.pool(x)
        f = self.pool(f)
        return o, x,f

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


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c,k,p, name=None):
        super(DecoderBlock, self).__init__()
        self.conv3 = nn.Conv2d(in_c,in_c,kernel_size=3,padding=1)
        self.r1 = ResidualBlock(in_c + in_c, out_c,k,p)
        self.r2 = ResidualBlock(out_c, out_c,k,p)
        self.bn = nn.BatchNorm2d(in_c)
        #self.se = SELayer(in_c*2,in_c*2)
        self.simam_module=simam()
        self.relu = nn.ReLU()

    def forward(self, inputs, skip):
        x = nn.functional.interpolate(inputs, scale_factor=2, mode='bilinear',align_corners=True)
        res = skip
        skip = self.relu(self.bn(self.conv3(skip)))
        skip = self.relu(self.bn(self.conv3(skip)))
        skip = self.simam_module(skip)
        skip = skip+res
        x = torch.cat([x, skip], axis=1)
        x = self.simam_module(x)
        x = self.r1(x)
        #x = self.r2(x)
        x = self.r2(x)
        return x
#ch 16 32 64 128
class SMSFBlock(nn.Module):
    def __init__(self,ch):
        super(SMSFBlock,self).__init__()
        self.conv3x3_s12 = nn.Conv2d(ch[0]+ch[1],ch[1],kernel_size=3,padding=1)
        self.conv3x3_s23 = nn.Conv2d(ch[1]+ch[2],ch[3],kernel_size=3,padding=1)
        self.conv3x3_s34 = nn.Conv2d(ch[2]+ch[3],ch[3],kernel_size=3,padding=1)
        self.in1 = nn.InstanceNorm2d(ch[1])
        self.in2 = nn.InstanceNorm2d(ch[2])
        self.in3 = nn.InstanceNorm2d(ch[3])
        self.relu = nn.ReLU(inplace=True)
        self.simam = simam()
        self.sconv1 = nn.Conv2d(ch[1] + ch[2], ch[2], kernel_size=3, padding=1)
        self.sconv2 = nn.Conv2d(ch[2] + ch[3], ch[2], kernel_size=3, padding=1)
        self.maxp = nn.MaxPool2d(2)

        self.oconv1 = nn.Conv2d(ch[0]+ch[2], ch[1], kernel_size=3, padding=1)
        self.oconv2 = nn.Conv2d(ch[2] + ch[2], ch[2], kernel_size=3, padding=1)
        self.oconv3 = nn.Conv2d(ch[2] + ch[3], ch[3], kernel_size=3, padding=1)
        self.oconv4 = nn.Conv2d(ch[2] + ch[3], ch[3]*2, kernel_size=3, padding=1)

    def forward(self,s1,s2,s3,s4):
        s1_s, s1_ =s1.chunk(2, dim=1)
        s2_s, s2_ = s2.chunk(2, dim=1)
        s3_s, s3_ = s3.chunk(2, dim=1)
        s4_s, s4_ = s4.chunk(2, dim=1)

        s2_s = nn.functional.interpolate(s2_s, scale_factor=2, mode='bilinear',align_corners=True)

        s12 = torch.cat([s1_, s2_s], dim=1)
        s12 = self.relu(self.in1(self.conv3x3_s12(s12)))
        s12 = self.maxp(s12)

        s3_s = nn.functional.interpolate(s3_s, scale_factor=2, mode='bilinear', align_corners=True)
        s23 = torch.cat([s2_,s3_s], dim=1)
        s23 = self.relu(self.in3(self.conv3x3_s23(s23)))

        s4_s = nn.functional.interpolate(s4_s, scale_factor=2, mode='bilinear', align_corners=True)
        s34 = torch.cat([s3_, s4_s], dim=1)
        s34 = self.relu(self.in3(self.conv3x3_s34(s34)))
        s34 = nn.functional.interpolate(s34, scale_factor=2, mode='bilinear',align_corners=True)

        s23_s, s23_ = s23.chunk(2, dim=1)

        sod = torch.cat([s12,s23_s], dim=1)
        sod = self.maxp(self.simam(self.sconv1(sod)))

        sou = torch.cat([s23_,s34], dim=1)
        sou = self.simam(self.sconv2(sou))

        sou_u = nn.functional.interpolate(sou, scale_factor=2, mode='bilinear', align_corners=True)
        os1 = torch.cat([s1_s, sou_u], dim=1)
        os1 = self.relu(self.oconv1(os1))

        os2 = torch.cat([sou,s2],dim=1)
        os2 = self.relu(self.oconv2(os2))

        os3 = torch.cat([s3,sod],dim=1)
        os3 = self.relu(self.oconv3(os3))

        sod_d = self.maxp(sod)
        os4 = torch.cat([s4_,sod_d],dim=1)
        os4 = self.relu(self.oconv4(os4))
        #print('os1  2  3 4',os1.shape,os2.shape,os3.shape,os4.shape)
        return os1,os2,os3,os4

class FANet(nn.Module):
    def __init__(self):
        super(FANet, self).__init__()
        self.e1 = EncoderBlock(3, 32,3,1,1)
        self.e2 = EncoderBlock(32, 64,3,1,8)
        self.e3 = EncoderBlock1(64, 128,3,1,16)
        self.e4 = EncoderBlock1(128, 256,3,1,32)

        self.SMSFBlock=SMSFBlock([16,32,64,128])

        self.d1 = DecoderBlock(256, 128,3,1)
        self.d2 = DecoderBlock(128, 64,3,1)
        self.d3 = DecoderBlock(64, 32,3,1)
        self.d4 = DecoderBlock(32, 16,3,1)

        self.output = nn.Conv2d(16, 1, kernel_size=1, padding=0)

    def forward(self, x):
        inputs, masks,ft = x[0], x[1],x[2]
        p1, s1,f1 = self.e1(inputs, masks,ft)
        p2, s2,f2 = self.e2(p1,masks,f1)
        p3, s3,f3 = self.e3(p2,f2)
        p4, s4,f4 = self.e4(p3,f3)
        #print("s1  2 3  4",s1.shape,s2.shape,s3.shape,s4.shape)
        s1,s2,s3,s4 = self.SMSFBlock(s1,s2,s3,s4)
        d1 = self.d1(p4, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        output = self.output(d4)
        return output
    
if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256)).cuda()
    m = torch.randn((2, 1, 256, 256)).cuda()
    f = torch.randn((2, 1, 256, 256)).cuda()
    model = FANet().cuda()
    y = model([x, m,f])
    print(y.shape)
