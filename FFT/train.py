
import os
import time
import datetime
import random
import numpy as np
from glob import glob
import albumentations as A
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import (
    seeding, shuffling, create_dir,
    epoch_time, rle_encode, rle_decode, print_and_save, load_data
    )
from FFTD import FFTDnet
from loss import DiceBCELoss,DiceLoss,BCELoss
from thop import profile

def show_feature_map(feature_map,i):
    feature_map=feature_map.cpu()
    feature_map=feature_map.detach().numpy().squeeze()
    feature_map_num = feature_map.shape[0]
    for index in range(feature_map_num):
        feature=feature_map[0][index]
        feature = np.asarray(feature*255,dtype=np.uint8)
        cv2.imwrite('/home/jn/FFTD--Net/Featuremap/img_{}_{}.png'.format(str(i),str(index)), feature)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = image.astype(np.float32)

        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = mask.astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples

def train(model, loader, optimizer, loss_fn,device):
    epoch_loss = 0
    model.train()
    LR=optimizer.state_dict()['param_groups'][0]['lr']
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        #b, c, h, w  = y.shape
        #m = []
        #for edata in mask[i*b : i*b+b]:
            #edata = " ".join(str(d) for d in edata)
            #edata = str(edata)
            #edata = rle_decode(edata, size)
            #edata = np.expand_dims(edata, axis=0)
            #m.append(edata)

        #m = np.array(m, dtype=np.int32)
        #m = np.transpose(m, (0, 1, 3, 2))
        #m = torch.from_numpy(m)
        #m = m.to(device, dtype=torch.float32)
        

        #flops,params =profile(model,(x,))
        #print(flops,params)
        optimizer.zero_grad()
        y_pred  = model(x)
        loss = loss_fn(y_pred, y)
        #print('The  ',i, '  it,  train loss : ',loss.item())
        loss.backward()
        optimizer.step()
        # if epoch%5==0:
        #     if i % 20 ==0:
        #         show_feature_map(f,i)
        print('\rTrain 进度为' ,i+1,'  lossDice:',loss.item(),end='')
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    print('')
    return epoch_loss,LR

def evaluate(model, loader, loss_fn,device):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            b, c, h, w  = y.shape
            
            #m = []
            #for edata in mask[i*b : i*b+b]:
            	#edata = " ".join(str(d) for d in edata)
            	#edata = str(edata)
            	#edata = rle_decode(edata, size)
            	#edata = np.expand_dims(edata, axis=0)
            	#m.append(edata)

            #m = np.array(m, dtype=np.int32)
            #m = np.transpose(m, (0, 1, 3, 2))
            #m = torch.from_numpy(m)
            #m = m.to(device, dtype=torch.float32)

            y_pred= model(x)
            loss = loss_fn(y_pred, y)
            print('\rval进度为' ,i+1,'  lossDice:',loss.item(),end='')
            epoch_loss += loss.item()

    print('')
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("filesM")

    """ Training logfile """
    train_log_path = "filesM/tlog17.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("filesM/tlog17.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)

    """ Hyperparameters """
    size = (256, 256)
    batch_size = 2
    num_epochs = 100
    lr = 1e-4
    checkpoint_path = "filesM/cp17.pth"

    """ Dataset """
    path = "/home/jn/Trans/ISIC2017"
    #path = "/home/jn/SKIN"
    #path = "/home/jn/FANet-main/Kvasir-Seg"
    (train_x, train_y), (valid_x, valid_y) = load_data(path)
    train_x, train_y= shuffling(train_x, train_y)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y,size, transform=None)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = FFTDnet()
    model.eval()
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,eta_min=5e-5,last_epoch=-1,verbose=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True,factor=0.5)
    loss_fn = DiceBCELoss()
    loss_name = "DiceBCE Loss"

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model. """
    best_valid_loss = float('inf')
    #GLCM = init_glcm(train_x, size)
    #val_GLCM = init_glcm(valid_x, size)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss,LR = train(model, train_loader,optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn,device)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            data_str = f"Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)
            torch.save(model.state_dict(), checkpoint_path)

            #train_mask = return_train_mask
            #valid_mask = return_valid_mask

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        tiiime = str(datetime.datetime.now())
        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s   time:{tiiime}\n'
        data_str += f'\tTrain Loss: {train_loss:.6f} \n'
        data_str += f'\t Val. Loss: {valid_loss:.6f} \n'
        data_str += f'\t Train LR : {LR}  \n'
        print_and_save(train_log_path, data_str)
