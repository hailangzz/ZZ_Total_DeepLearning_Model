# torchvision模块实战
# torchvision.datasets模块包括数据集，数据加载方法
# torchvision.models模块包括一些经典的网络架构
# torchvision.transforms模块包括一些预处理图像增强方法
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


# 处理流程

# 数据预处理部分：
# 数据增强：torchvision.transforms
# 数据预处理：torchvisivon.transforms
# DataLoader读取batch数据

# 数据读取与预处理操作
data_dir = r'D:\迅雷下载\AI数据集汇总\flower_data'
train_dir = data_dir + 'train'
valid_dir = data_dir + 'valid'

# data_transforms中指定了所有图像的预处理操作
# ImageFolder假设所有文件按文件夹保存好，每个文件夹下面存储同一类别图片，文件夹的名字为分类的名字
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
batch_size = 8  # 显存不够把batch_size调小
# 构建数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train', 'valid']}  # 传路径和预处理流程
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}
datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
print(class_names)
print(dataloaders)
print(datasets_sizes)

# 读取标签和对应的实际名字
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)


# 展示数据
# 因为我们已经对数据做了处理，所以先将tensor数据转化为numpy格式，然后还原回标准化的结果
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)  # torch颜色通道被放到了第一位，我们利用transpose将h,w,c还原回去
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # 去除标准化，成标准差加均值
    image = image.clip(0, 1)

    return image


fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2

dataiter = iter(dataloaders['valid'])  # 迭代一次取一个batch数据
inputs, classes = dataiter.next()  # 数据，标签

'''
for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    plt.imshow(im_convert(inputs[idx]))
plt.show()
'''

# 迁移学习
# 拿别人的卷积层，方案1在此卷积层的基础上继续训练，方案2冻结此卷积层作为我们的特征提取工具
# 全连接层都是要重写重训练的

# 网络模块设置：
# 加载预训练模型：torchvision.models，可以调用训练好的权重参数来继续训练，也就是所谓的迁移学习
# 注意：需要把最后一层改一下，改成我们自己的任务
# 训练时可以全部重头训练，也可以只训练我们的任务层，因为前几层都是做特征提取的，本质任务目标是一致的

model_name = 'resnet'  # 可选择的模型比较多{'alexnet', 'vgg', 'resnet', 'squeezenet', 'densenet', 'inception'}
# 是否用人家训练好的特征来做
feature_extract = True

# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# 加载模型
model_ft = models.resnet50()
print(model_ft)


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


num_classes = len(class_names)
print('\n\n\n\n\n\n\n\ntrain calssify numbers:%s \n\n\n'%num_classes)
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# GPU计算
model_ft = model_ft.to(device)

# 指定保存模型的名字
filename = 'checkpoint.pth'

# 是否训练所有层,一般策略是先学习自己的层，在观察学习全部层
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# 看一下现在的网络层
print(model_ft)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)  # 学习率每7个epoch衰减为原来的0.1
# 最后一层LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=filename):
    since = time.time()
    best_acc = 0
    """
    """
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)  # 获取概率最大的类别

                    # 训练更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好次好的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)

            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
                scheduler.step(epoch_loss)

        lr_train_values = optimizer.param_group[0]['lr']
        print('Optimizer learning rate : {:.7f}'.format(lr_train_values))
        LRs.append(lr_train_values)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当作模型最终结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer_ft,
                                                                                            num_epochs=10,
                                                                                            is_inception=False,
                                                                                            filename=filename)
# 模型保存可以带有选择性，例如在验证集中如果效果好则保存


