import os
import cv2
# def Edge_Extract(root):
# 	img_root='./ISIC/mask'
# 	edge_root='./ISIC/edge'
# 	if not os.path.exists(edge_root):
# 		os.mkdir(edge_root)
# 	file_names = os.listdir(img_root)
# 	img_name=[]
# 	for name in file_names:
# 		if not name.endswith('.png'):
# 			assert "this is no PNG"
# 		img_name.append((os.path.join(img_root,name[:-4]+'.png')))
# 	index =0
# 	for image in img_name:
# 		img =cv2.imread(image,0)
# 		cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,30,100))
# 		index +=1
# 	return 0
#
# if __name__=='__main__':
# 	root='..'
# 	Edge_Extract(root)
import random
path='/home/jn/Trans/ISIC/image/'
# train_pr=0.9
files=os.listdir(path)
num=0
for f in files:
	if f.find('jpg') !=-1:
		imgp = r'/home/jn/Trans/ISIC/image/'+f
		img = cv2.imread(imgp)
		if img.shape[0]>2600 :
			print(img.shape)
			num = num+1
print(num)

# path17='/home/jn/Trans/ISIC2017/image/'
# # train_pr=0.9
# files17=os.listdir(path17)
# print(len(path),len(path17))
# for i in files:
# 	if i not in files17:
# 		print(i)
# # num=len(files)
# list=range(num)
# train=random.sample(list,int(num*0.9))
# traintxt=open('/home/jn/Trans/ISIC/train6.txt','w')
# valtxt=open('/home/jn/Trans/ISIC/val6.txt','w')
# #traintxt=open('/home/jn/FANet-main/DSB/test.txt','w')
# for i in list:
# 	if i in train:
# 		traintxt.write(files[i].split('.')[0]+'\n')
# 	else:
# 		valtxt.write(files[i].split('.')[0]+'\n')
