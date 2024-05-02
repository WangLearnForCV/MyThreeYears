import os

import torch
import cv2
from FFTD import FFTDnet
import numpy as np
checkpoint_path='./filesCross/cpISICc.pth'
device = torch.device('cuda')
model = FFTDnet()
model.eval()
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
#file = open('/home/jn/FANet-main/DSB/Heat.txt')
file = open('/home/jn/Trans/ISIC/Heat.txt')
ft = file.readlines()
names = []
for im in ft:
    names.append(im.split('\n')[0])
for imgp in names:
    #img = cv2.imread('/home/jn/FANet-main/DSB/image/'+imgp+'.png',cv2.IMREAD_COLOR)
    img = cv2.imread('/home/jn/Trans/ISIC/image/' + imgp + '.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    y_pred ,d = model(img.to(device))
    #d = torch.sigmoid(d)
    # 1.1 获取feature maps
    features = d  # 尺度大小，如：torch.Size([1,80,45,45])
    # 1.2 每个通道对应元素求和
    heatmap = torch.sum(features, dim=1)  # 尺度大小， 如torch.Size([1,45,45])
    max_value = torch.max(heatmap)
    min_value = torch.min(heatmap)
    heatmap = (heatmap-min_value)/(max_value-min_value)*255
    heatmap = heatmap.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0)  # 尺寸大小，如：(45, 45, 1)
    src_size = (256,256)  # 原图尺寸大小
    heatmap = cv2.resize(heatmap, src_size,interpolation=cv2.INTER_LINEAR)  # 重整图片到原尺寸
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    # 保存热力图
    cv2.imwrite('./Featuremap/Skip_'+imgp+'.jpg', heatmap)

