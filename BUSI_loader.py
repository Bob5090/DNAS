import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
# 类初始化
    def __init__(self, root):
        self.imgs_path = root
# 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        return img_path
# 返回长度
    def __len__(self):
        return len(self.imgs_path)

# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r'BUSI2019/images/*/*.png')
# 循环遍历输出列表中的每个元素，显示出每个图片的路径
# for var in all_imgs_path:
#     print(var)


# 利用自定义类Mydataset创建对象weather_dataset
# BUSI_dataset = Mydataset(all_imgs_path)
# print(len(BUSI_dataset)) #返回文件夹中图片总个数
# print(BUSI_dataset[12:14])#切片，显示第12张、第十三张图片，python左闭右开
# BUSI_datalodaer = torch.utils.data.DataLoader(BUSI_dataset, batch_size=3) #每次迭代时返回五个数据
# print(next(iter(BUSI_datalodaer)))

species = ['benign', 'malignant', 'normal']
species_to_id = dict((c, i) for i, c in enumerate(species))
#print(species_to_id)
id_to_species = dict((v, k) for k, v in species_to_id.items())
#print(id_to_species)
all_labels = []
#对所有图片路径进行迭代
for img in all_imgs_path:
    # 区分出每个img，应该属于什么类别
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)
#print(all_labels) #得到所有标签

size = 256
# 对数据进行转换处理
transform = transforms.Compose([
                transforms.Resize((size, size)), #做的第一步转换
                transforms.ToTensor() #第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
])

class Mydatasetpro(data.Dataset):
# 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform
# 进行切片
    def __getitem__(self, index):                #根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)                 #pip install pillow
        data = self.transforms(pil_img)
        return data, label
# 返回长度
    def __len__(self):
        return len(self.imgs)



# if __name__ == '__main__':

BATCH_SIZE = 16
BUSI_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)
BUSI_datalodaer = data.DataLoader(
                            BUSI_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True
)

print(BUSI_datalodaer.dataset)

imgs_batch, labels_batch = next(iter(BUSI_datalodaer))
#print(imgs_batch.shape)

# plt.figure(figsize=(12, 8))
# for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
#     img = img.permute(1, 2, 0).numpy()
#     plt.subplot(2, 3, i+1)
#     plt.title(id_to_species.get(label.item()))
#     plt.imshow(img)
# plt.show()
#
#划分测试集和训练集
index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]

#80% as train
s = int(len(all_imgs_path)*0.8)
v = int(len(all_imgs_path)*0.2)
# print(s)
# print(v)
#
train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
# valid_imgs = all_imgs_path[s:s+v]
# valid_labels = all_labels[s:s+v]
test_imgs = all_imgs_path[s:]
test_labels = all_labels[s:]
print(len(train_imgs),len(train_labels))
# print(len(valid_imgs),len(valid_labels))
print(len(test_imgs),len(test_labels))

train_set = Mydatasetpro(train_imgs, train_labels, transform) #TrainSet TensorData
# valid_ds = Mydatasetpro(valid_imgs, valid_labels, transform) #TestSet TensorData
test_set = Mydatasetpro(test_imgs, test_labels, transform) #TestSet TensorData

train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)#TrainSet Labels
# valid_dl = data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)#TestSet Labels
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)#TestSet Labels

# print(train_dl.dataset)
