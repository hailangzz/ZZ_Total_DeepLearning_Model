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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 加载训练好的模型
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU模式
model_ft = model_ft.to(device)

# 保存文件的名字
filename = r'D:\PycharmProgram\total_model\checkpoint.pth'

# 加载模型
checkpoint = torch.load(filename,map_location=torch.device('cpu'))
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

    return torch.tensor(img,dtype=torch.float)

image_path = r'D:\迅雷下载\AI数据集汇总\flower_data\train\88\image_00448.jpg'
img = process_image(image_path)


model_ft.eval()

img=img.unsqueeze(0)
output = model_ft(img)

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
print(preds)
