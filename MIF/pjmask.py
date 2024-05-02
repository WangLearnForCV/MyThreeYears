import cv2
import os
import shutil
import numpy as np

def fusionimg(mask_path,id):
    imgnumpy = []
    for path in mask_path:
        #shutil.copy(path,'./DataSB/')
        im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        imgnumpy.append(im)
    h,w = imgnumpy[0].shape[0],imgnumpy[0].shape[1]
    init=np.zeros((h,w))
    for nps in imgnumpy:
        init+=nps
    cv2.imwrite('/home/jn/FANet-main/DSB/mask/'+id+'.png',init)
    cv2.waitKey(0)

ids = os.listdir('./DataSB/data-science-bowl-2018/stage1_train/')
for id in ids:
    path = './DataSB/data-science-bowl-2018/stage1_train/'+id+'/masks/'
    masks = os.listdir(path)
    mask_path = []
    for mask in masks:
        mask_path.append(path+mask)
    fusionimg(mask_path,id)
    im= './DataSB/data-science-bowl-2018/stage1_train/'+id+'/images/'+id+'.png'
    shutil.copy(im, '/home/jn/FANet-main/DSB/image/')
