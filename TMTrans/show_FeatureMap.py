import torch
import cv2
from Ttran import TWLNet
import numpy as np
import fast_glcm

checkpoint_path='./filessch/cpDSB.pth'
device = torch.device('cuda')
model = TWLNet()
model.eval()
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
imgp='76a372bfd3fad3ea30cb163b560e52607a8281f5b042484c3a0fc6d0aa5a7450'
# imgp='ISIC_0000191'
img = cv2.imread('/home/jn/FANet-main/DSB/image/'+imgp+'.png',cv2.IMREAD_COLOR)
cpimg = img
img = cv2.resize(img,(256,256))
img = np.transpose(img, (2, 0, 1))
img = torch.from_numpy(img).float()
img = img.unsqueeze(0)


def GLCM(image, size):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    glcm_mean = fast_glcm.fast_glcm_mean(img)
    return glcm_mean

def rle_encode(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b

    return run_lengths
glcm = GLCM('/home/jn/FANet-main/DSB/image/'+imgp+'.png' ,(256,256))

glcm = torch.from_numpy(glcm).float()
glcm = glcm.unsqueeze(0)
glcm = glcm.unsqueeze(0)
y_pred ,d = model([img.to(device),glcm.to(device)])
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
cv2.imwrite('./FeatMap/heapmap_'+imgp+'.jpg', heatmap)
