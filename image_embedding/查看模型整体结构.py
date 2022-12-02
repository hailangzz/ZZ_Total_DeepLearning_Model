import torch
import torchvision.models as models
import numpy as np


# 参数pretrained默认为False,意味着得到一个随机初始化参数的vgg19模型。
# vgg_model = models.vgg19()
# 可以通过提供本地参数文件的路径, 并使用load_state_dict加载，
# 得到一个参数是预训练好的vgg19模型。
# pre_file = torch.load('/XXXX/vgg19-dcbb9e9d.pth')
# vgg_model.load_state_dict(pre_file)

# 如果将pretrained设置为True, 意味着直接得到一个加载了预训练参数的vgg19模型。
# 就会自动下载vgg19的参数文件并放在本地缓存中。所以不用提供本地参数文件的路径。
vgg_model = models.vgg19(pretrained=True)

# 查看模型整体结构
structure = torch.nn.Sequential(*list(vgg_model.children())[:])
print(structure)

# 查看模型各部分名称
print('模型各部分名称', vgg_model._modules.keys())

# 获取vgg19模型的第一个Sequential, 也就是features部分.
features = torch.nn.Sequential(*list(vgg_model.children())[0])
print('features of vgg19: ', features)

# 获取vgg19模型的最后一个Sequential, 也就是classifier部分.
classifier = torch.nn.Sequential(*list(vgg_model.children())[-1])
print('classifier of vgg19: ', classifier)

# 在获取到最后一个classifier部分的基础上, 再切割模型, 去掉最后一层.
new_classifier = torch.nn.Sequential(*list(vgg_model.children())[-1][:6])
print('new_classifier: ', new_classifier)

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 获取vgg19原始模型, 输出图像维度是1000.
vgg_model_1000 = models.vgg19(pretrained=True)

# 下面三行代码功能是:得到修改后的vgg19模型.
# 具体实现是: 去掉vgg19原始模型的第三部分classifier的最后一个全连接层,
# 用新的分类器替换原始vgg19的分类器，使输出维度是4096.
vgg_model_4096 = models.vgg19(pretrained=True)
new_classifier = torch.nn.Sequential(*list(vgg_model_4096.children())[-1][:6])
vgg_model_4096.classifier = new_classifier

# 获取和处理图像
image_dir = r'D:\迅雷下载\AI数据集汇总\猫狗图像识别\dogs-vs-cats\test\test\2.jpg'
im = Image.open(image_dir)
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
im = trans(im)
im.unsqueeze_(dim=0)

# 使用vgg19得到图像特征.
# 原始vgg19模型
image_feature_1000 = vgg_model_1000(im).data[0]
print('dim of vgg_model_1000: ', image_feature_1000.shape)

# 修改后的vgg19模型
image_feature_4096 = vgg_model_4096(im).data[0]
print('dim of vgg_model_4096: ', image_feature_4096.size())
print(np.array(image_feature_4096))