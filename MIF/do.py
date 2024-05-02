import os
path='/home/jn/FANet-main/ISIC/image/image'
files=os.listdir(path)
f=open('train.txt',mode='w')
for file in files:
    if '.jpg' in files:
    	f.writelines(file.split('.')[0])
    	print(file.split('.')[0])
