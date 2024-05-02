文件夹1对应的是论文中第一篇论文：用于黑色素瘤分割的多信息融合的注意力算法 MIF
（以下适用于所有算法的train.py与test.py）
运行train.py: pycharm直接点击train.py 运行；命令行 在train.py文件夹下 使用 python train.py
train_log_path 为日志存放文件
训练时，在train.py文件中，需要注意以下参数
size(,)用于设置图像大小
batch_size 为批次大小
num_epochs 为训练轮次
lr = 1e-4  初始化学习率
checkpoint_path 为训练权重存放文件
path 记录数据集文件存放位置
注：
数据集格式
 --ISIC
	--image
	--mask
	--train.txt
	--test.txt
如需要在已有模型上继续训练 请将以下代码取消注释：
#model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  优化器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)  学习策略
loss_fn = DiceBCELoss()  损失函数

test.py中
运行test.py: pycharm直接点击test.py 运行；命令行 在test.py文件夹下 使用 python test.py
分割结果在create_dir()所创建的文件（最后一行  cv2.imwrite(f"@/{name}.png", cat_images))  @需要与create_dir()所创建的文件路径一致

utils.py中
load_data()方法下的  images与masks需要根据具体数据集进行后缀的更改

文件夹2对应的是论文中第二篇论文：用于黑色素瘤分割的混合纹理特征的Transformer算法  TMTrans
文件夹3对应的是论文中第三篇论文：引入频域特征的注意力机制的黑色素瘤分割算法 FFT

运行与配置同上

运行环境及需要的环境依赖包：
albumentations
torch
cv2
numpy
tqdm
glob
python 3.8
cuda 10.1