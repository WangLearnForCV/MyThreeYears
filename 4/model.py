import torch
import torch.nn as nn
from blocks import TCIEncoder, TCIDecoder, CIBlock, DFGBlock
import cv2
import numpy as np
class TCINet(nn.Module):
    def __init__(self):
        super(TCINet, self).__init__()
        self.e1 = TCIEncoder(3, 32)
        self.e2 = TCIEncoder(32, 64)
        self.e3 = TCIEncoder(64, 128)
        self.e4 = TCIEncoder(128, 256)

        self.CIBlock = CIBlock([32, 64, 128])
        self.DFGBlock = DFGBlock()

        self.d4 = TCIDecoder(256, 128)
        self.d3 = TCIDecoder(128, 64)
        self.d2 = TCIDecoder(64, 32)
        self.d1 = TCIDecoder(32, 16)

        self.maxpool = nn.MaxPool2d(2)
        self.outconv = nn.Conv2d(16+3, 1, kernel_size=1)

    def forward(self, x):
        dog = self.DFGBlock(x)
        x1, s1 = self.e1(x)
        #print('x and skip  ', x1.shape,s1.shape)
        x1 = self.maxpool(x1)
        x2, s2 = self.e2(x1)
        x2 = self.maxpool(x2)
        #print('x and skip  ',x2.shape, s2.shape)
        x3, s3 = self.e3(x2)
        x3 = self.maxpool(x3)
        #print('x and skip  ',x3.shape, s3.shape)
        x4, s4 = self.e4(x3)
        x4 = self.maxpool(x4)

        #print('x and skip  ',x4.shape, s4.shape)
        s2, s3, s4 = self.CIBlock(s2, s3, s4)

        d4 = self.d4(x4, s4, True)
        d3 = self.d3(d4, s3, True)
        d2 = self.d2(d3, s2, True)
        d1 = self.d1(d2, s1, True)
        d1 = torch.cat([d1, dog], dim=1)
        out = self.outconv(d1)

        return out
    
if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256)).cuda()
    model = TCINet().cuda()
    y = model(x)
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)#18.64
    print('      - Params: ' + params)#7.04
    print(y.shape)
