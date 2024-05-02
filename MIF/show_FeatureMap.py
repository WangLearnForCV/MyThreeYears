import torch
import cv2
from model import FANet
import numpy as np

checkpoint_path='./files/checkpoint.pth'
device = torch.device('cuda')
model = FANet()
model.eval()
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
imgp='eb96fc6cbf6880bf05c4309857ae33844a4bc2152e228eff31024e5265cf9fc3'
# imgp='ISIC_0000191'
img = cv2.imread('/home/jn/FANet-main/DSB/image/'+imgp+'.png',cv2.IMREAD_COLOR)
cpimg = img
img = cv2.resize(img,(256,256))
img = np.transpose(img, (2, 0, 1))
img = torch.from_numpy(img).float()
img = img.unsqueeze(0)


def OTSU(image, size):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = th.astype(np.int32)
    th = th / 255.0
    th = th > 0.5
    th = th.astype(np.int32)
    return img, th

def FT(image,size):
    # print(type(image))
    img = cv2.imread(image)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gray_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l_mean = np.mean(gray_lab[:, :, 0])
    a_mean = np.mean(gray_lab[:, :, 1])
    b_mean = np.mean(gray_lab[:, :, 2])
    lab = np.square(gray_lab - np.array([l_mean, a_mean, b_mean]))
    lab = np.sum(lab, axis=2)
    lab = lab / np.max(lab)
    return lab

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
mask,pp = OTSU('/home/jn/FANet-main/DSB/image/'+imgp+'.png' ,(256,256))
#mask,pp = OTSU('/home/jn/FANet-main/DSB/image/'+imgp+'.jpg', (256,256))
mask = torch.from_numpy(mask).float()
mask = mask.unsqueeze(0)
mask = mask.unsqueeze(0)

FT = FT('/home/jn/FANet-main/DSB/image/'+imgp+'.png' ,(256,256))
#FT = FT('/home/jn/Trans/FANet-main/DSB/'+imgp+'.jpg' ,(256,256))
FT = torch.from_numpy(FT).float()
FT = FT.unsqueeze(0)
FT = FT.unsqueeze(0)

y_pred ,d = model([img.to(device),mask.to(device),FT.to(device)])
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