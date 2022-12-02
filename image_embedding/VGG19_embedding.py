import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pickle
from numpy import linalg as LA

# 获取vgg19原始模型, 输出图像维度是1000.
vgg_model_1000 = models.vgg19(pretrained=True)

# 下面三行代码功能是:得到修改后的vgg19模型.
# 具体实现是: 去掉vgg19原始模型的第三部分classifier的最后一个全连接层,
# 用新的分类器替换原始vgg19的分类器，使输出维度是4096.
vgg_model_4096 = models.vgg19(pretrained=True)



new_classifier = torch.nn.Sequential(*list(vgg_model_4096.children())[-1][:6])
print(new_classifier)
vgg_model_4096.classifier = new_classifier

# 获取和处理图像
def creat_image_matrix(image_path):
    # im = Image.open(image_path) #读取png格式图片出问题
    im = Image.open(image_path).convert('RGB')
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    im = trans(im)
    im.unsqueeze_(dim=0)
    return im

# # 使用vgg19得到图像特征.
# # 原始vgg19模型
# image_feature_1000 = vgg_model_1000(im).data[0]
# print('dim of vgg_model_1000: ', image_feature_1000.shape)
#
# # 修改后的vgg19模型
# image_feature_4096 = vgg_model_4096(im).data[0]
# print('dim of vgg_model_4096: ', image_feature_4096.shape)


origin_image_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\test'
def get_images_embedding(total_images_name,origin_image_path):

    total_images_feature_embedding_dict={}
    number = 0
    for single_image_name in total_images_name:
        number+=1
        print(number)
        single_image_path = os.path.join(origin_image_path,single_image_name)
        image_matrix = creat_image_matrix(single_image_path)
        predict_matrix = vgg_model_4096(image_matrix).data[0]
        normal_matrix = predict_matrix/LA.norm(predict_matrix)
        # print(normal_matrix)
        image_feature_embedding = np.array(normal_matrix)

        if single_image_name not in total_images_feature_embedding_dict:
            total_images_feature_embedding_dict[single_image_name] = image_feature_embedding

    return total_images_feature_embedding_dict


total_images_name = os.listdir(origin_image_path)

total_image_embedding = get_images_embedding(total_images_name,origin_image_path)

pickle.dump(total_image_embedding, open(os.path.join(origin_image_path,'total_image_embedding.p'), 'wb'))

# 读取total_image_embedding 存储信息：
total_image_embedding_pickle = pickle.load(open(os.path.join(origin_image_path,'total_image_embedding.p'), mode='rb'))
print(total_image_embedding_pickle)





