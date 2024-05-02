import os
import random
path='/home/jn/SKIN/image'
train_pr=0.9
files=os.listdir(path)
num=len(files)
list=range(num)
train=random.sample(list,int(num*0.9))
traintxt=open('/home/jn/SKIN/train.txt','w')
valtxt=open('/home/jn/SKIN/val.txt','w')
#traintxt=open('/home/jn/FANet-main/DSB/test.txt','w')
for i in list:
	if i in train:
		traintxt.write(files[i].split('.')[0]+'\n')
	else:
		valtxt.write(files[i].split('.')[0]+'\n')
