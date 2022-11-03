import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
# import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同的模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet152
        """
        model_ft = models.resnet50(pretrained=use_pretrained)  # 下载预训练的模型参数
        set_parameter_requires_grad(model_ft, feature_extract)  # 冻住卷积层
        num_ftrs = model_ft.fc.in_features  # 拿到最后全连接层输入
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.LogSoftmax(dim=1))  # 重写全连接层

        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

model_name = 'resnet'  # 可选择的模型比较多{'alexnet', 'vgg', 'resnet', 'squeezenet', 'densenet', 'inception'}
feature_extract = True
# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU模式
model_ft = model_ft.to(device)

# 保存文件的名字
filename = 'checkpoint.pth'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])


# 测试数据预处理
def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize，thumbnail方法只能进行缩小，所以进行判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # 裁剪操作
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # 相同的归一化，正则化
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # 注意颜色通道放到第一个位置
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()

    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax


image_path = 'D:\迅雷下载\AI数据集汇总/flower_data/flower_data/test/1/image_05087.jpg'
img = process_image(image_path)
imshow(img)
print(img.shape)

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机旋转
                                 transforms.CenterCrop(224),  # 从中心开始裁剪
                                 transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个翻转概率
                                 transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                                 transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                 # 亮度，对比度，饱和度，色相
                                 transforms.RandomGrayscale(p=0.025),  # 概率转换为灰度率
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 # 标准化((x-均值)/标准差)，均值，标准差，拿别人的预训练数据的均值和标准差为了使训练效果更好
                                 ]),
    'valid': transforms.Compose([transforms.Resize(256),  # 我们的训练集比较小所以没有resize
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 大小和标准化必须和训练集一样
                                 ]),
}
data_dir = r'D:\迅雷下载\AI数据集汇总\flower_data'
batch_size = 8  # 显存不够把batch_size调小
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train', 'valid']}  # 传路径和预处理流程
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}

# 得到一个batch的测试数据
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.next()

model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

print(output.shape)
# torch.Size([8, 102])

# 得到概率最大的那个
_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
print(preds)

# 预测结果展示
fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
    color = ("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))

plt.show()
