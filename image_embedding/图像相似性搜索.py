import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.utils import image_utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from scipy import spatial
import cv2
import pickle
from shutil import copy
from tqdm import tqdm

class Image_Similar_Class():

    def __init__(self):
        self.available_models = ['vgg16', 'resnet50']
        self.model_name = 'vgg16'
        self.model = 'load_model'

    def load_model(self,model_name, include_top=True):
        """ Load pre-trained Keras model
        Args:
            model_name: String, name of model to load
            include_top: String, the model is buildt with 'feature learning block' + 'classification block'
        Returns:
            model: Keras model instance
        """
        if self.model_name in self.available_models:
            # Load a Keras instance
            try:
                if model_name == 'vgg16':
                    model = VGG16(weights='imagenet', include_top=include_top)
                elif model_name == 'resnet50':
                    model = ResNet50(weights='imagenet', include_top=include_top)
                print(f">> '{model.name}' model successfully loaded!")
            except:
                print(f">> Error while loading model '{self.selected_model}'")

        # Wrong selected model
        else:
            print(f">> Error: there is no '{self.selected_model}' in {self.available_models}")

        self.model = model
        return model

    def get_img_size_model(self): #设置模型对应的，图像矩阵输入尺寸···
        """Returns image size for image processing to be used in the model
        Args:
            model: Keras model instance
        Returns:
            img_size_model: Tuple of integers, image size
        """
        model_name = self.model.name
        if model_name == "vgg16":
            img_size_model = (224, 224)
        elif model_name == "resnet50":
            img_size_model = (224, 224)
        else:
            img_size_model = (224, 224)
            print("Warning: model name unknown. Default image size: {}".format(img_size_model))

        return img_size_model

    def get_layername_feature_extraction(self): #设置神经网络模型对应的特征输出层
        """ Return the name of last layer for feature extraction
        Args:
            model: Keras model instance
        Returns:
            layername_feature_extraction: String, name of the layer for feature extraction
        """
        model_name = self.model.name
        if model_name == "vgg16":
            layername_feature_extraction = 'fc2'
        elif model_name == "resnet50":
            layername_feature_extraction = 'predictions'
        else:
            layername_feature_extraction = ''
            print("Warning: model name unknown. Default layername: '{}'".format(layername_feature_extraction))

        return layername_feature_extraction

    # 图像矩阵输入预处理
    def image_processing(self,img_array):
        """ Preprocess image to be used in a keras model instance
        Args:
            img_array: Numpy array of an image which will be predicte
        Returns:
            processed_img = Numpy array which represents the processed image
        """
        # Expand the shape
        img = np.expand_dims(img_array, axis=0)

        # Convert image from RGB to BGR (each color channel is zero-centered with respect to the ImageNet dataset, without scaling)
        processed_img = preprocess_input(img)

        return processed_img

    def get_feature_vector(self,img_path): #计算图像在指定模型下的特征输出矩阵值

        try:
            """ Get a feature vector extraction from an image by using a keras model instance
            Args:
                model: Keras model instance used to do the classification.
                img_path: String to the image path which will be predicted
            Returns:
                feature_vect: List of visual feature from the input image
            """

            # Creation of a new keras model instance without the last layer
            layername_feature_extraction = self.get_layername_feature_extraction()  # 获取模型特征提取层的名称：
            model_feature_vect = Model(inputs=self.model.input,
                                       outputs=self.model.get_layer(layername_feature_extraction).output)  # 计算输入下的模型提取层的特征向量结果

            # Image processing
            img_size_model = self.get_img_size_model()  # 获取模型需要的输入矩阵尺寸
            img = image_utils.load_img(img_path, target_size=img_size_model)  # 读取图片路径下的指定尺寸图像矩阵
            img_arr = np.array(img)  # 矩阵格式转换
            img_ = self.image_processing(img_arr)  # 图像矩阵的中心标准化、通道转换

            # Visual feature extraction
            feature_vect = model_feature_vect.predict(img_)  # 模型提取图像矩阵的特征向量

            return feature_vect

        except Exception as e:
            print(e)

    # 计算特征矩阵之间的余弦相似度值···
    def calculate_similarity(self,image_path_a, image_path_b):
        """Compute similarities between two images using 'cosine similarities'
        Args:
            vector1: Numpy vector to represent feature extracted vector from image 1
            vector2: Numpy vector to represent feature extracted vector from image 1
        Returns:
            sim_cos: Float to describe the similarity between both images
        """
        vector1 = self.get_feature_vector(image_path_a)
        vector2 = self.get_feature_vector(image_path_b)
        sim_cos = 1 - spatial.distance.cosine(vector1, vector2)

        return sim_cos


class Factory_Compare_Image_Info(): # 批量化比较图片相似度排序信息

    def __init__(self):
        self.origin_image_path=''
        self.father_path_images=''
        self.total_images_feature_embedding_dict = {}
        pass

    def get_images_simcos(self,image_similar_boject,origin_image_path):
        self.origin_image_path = origin_image_path
        stand_image_path = ''
        # 获取图像集存储路径下，所有图像的特征提取字典信息

        total_images_name = os.listdir(origin_image_path)
        number = 0
        for i in tqdm(range(len(total_images_name))):
            single_image_name = total_images_name[i]
            if number==0:
                stand_image_path = os.path.join(origin_image_path, single_image_name)

            single_image_path = os.path.join(origin_image_path, single_image_name)
            # 计算图片路径下的图像特征矩阵：
            sim_cos = image_similar_boject.calculate_similarity(stand_image_path,single_image_path)
            print('sim_cos :',sim_cos)
            if single_image_name not in self.total_images_feature_embedding_dict:
                self.total_images_feature_embedding_dict[single_image_name] = sim_cos

            number += 1

        # 获取图片集的父目录：
        self.father_path_images = os.path.dirname(origin_image_path)
        # pickle 记录下图片集的特征矩阵信息
        write_cur = open(os.path.join(self.father_path_images, 'images_simcos.txt'),'w')
        for key in self.total_images_feature_embedding_dict:
            write_cur.write(key+'   '+str(self.total_images_feature_embedding_dict[key]))
            write_cur.write('\n')
        write_cur.close()
        pickle.dump(self.total_images_feature_embedding_dict, open(os.path.join(self.father_path_images, 'total_image_embedding.p'), 'wb'))

    def save_sort_image_with_simcos(self,sort_image_save_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\sort_total_image_save2'):

        # 读取total_image_embedding 存储信息：
        total_image_simcos_pickle = pickle.load(open(os.path.join(os.path.dirname(origin_image_path), 'total_image_embedding.p'), mode='rb'))
        # print(total_image_simcos_pickle.keys(),total_image_simcos_pickle.values())
        images_simcos_sort_list = np.argsort(list(total_image_simcos_pickle.values()))

        for index in range(len(images_simcos_sort_list)):
            image_name = list(total_image_simcos_pickle.keys())[images_simcos_sort_list[index]]
            # print(os.path.join(origin_image_path, image_name))
            copy(os.path.join(origin_image_path, image_name), sort_image_save_path + '\\' + str(index) + '.png')



model_name = 'vgg16'
imag_similar = Image_Similar_Class()
imag_similar.load_model(model_name)
print(imag_similar.model_name)

# 计算单独两张图片的相似度
image_path_a = r'C:\Users\34426\Pictures\psc 95).jpg'
image_path_b = r'C:\Users\34426\Pictures\psc 2(5).jpg'
sim_cos = imag_similar.calculate_similarity(image_path_a,image_path_b)
print(sim_cos)

# 批量计算图片集相似度
factory_compare_image = Factory_Compare_Image_Info()
origin_image_path = r'D:\迅雷下载\AI数据集汇总\红绿灯检测数据集\traffic_light_detection\traffic_light_detection\new_mask_sample_CircularBackground\sort_total_image_save'
factory_compare_image.get_images_simcos(imag_similar,origin_image_path)
factory_compare_image.save_sort_image_with_simcos()