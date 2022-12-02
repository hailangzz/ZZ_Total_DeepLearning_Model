import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image


def create_folder(folder_name):
    """ Create folder if there is not
    Args:
        folder_name: String, folder name
    Returns:
        None
    """
    if not os.path.isdir(f"../models/{folder_name}"):
        os.makedirs(f"../models/{folder_name}")
        print(f"Folder '../models/{folder_name}' created")

def get_augmented_img_path_list(img_path, img_generator, nb_img):
    """ Performs a data augmentation
    Args:
        img_path: String to the image path
        img_generator: Tensor which generate batches of tensor image data with real-time data augmentation #
        nb_img: Integer, number of augmented images #增强生成多少张图像
    Returns:
        aug_img_list: Numpy array of augmented images
    """
    dir_aug_img = "../report/augmented_img"
    create_folder(dir_aug_img)

    # Read img
    img = plt.imread(img_path)
    filename = os.path.basename(img_path).split(".")[0]
    img_ = np.expand_dims(img, 0)  # 图像矩阵增加一个维度，方便后续处理

    # Generate batches of augmented images from the original image
    aug_iter = gen.flow(img_)  # ImageDataGenerator 图片数据增强处理管道

    # Get nb_img samples of augmented images 获取数据增强后的图片
    aug_img = [next(aug_iter)[0].astype(np.uint8) for i in range(nb_img)]

    aug_img_path_list = []
    for i in range(len(aug_img)):
        # Save augmented images
        new_filename = f"{filename}AI{i}".format(filename, i)
        aug_img_path = f"{dir_aug_img}/{new_filename}.jpg"
        aug_img_to_save = Image.fromarray(aug_img[i])
        aug_img_to_save.save(aug_img_path)

        # Add augmented images
        aug_img_path_list.append(aug_img_path)

    return aug_img_path_list

# Generate batches of tensor image data with real-time data augmentation.
gen = ImageDataGenerator(
    rotation_range=30, # Int: degree range for random rotations
    width_shift_range=0.1, # Float: fraction of total width, if < 1, or pixels if >= 1
    height_shift_range=0.1, # Float: fraction of total height, if < 1, or pixels if >= 1
    shear_range=0.15, # Float: shear Intensity (shear angle in counter-clockwise direction in degrees)
    zoom_range=0.1, # Float: range for random zoom
    channel_shift_range=10., # Float: range for random channel shifts
    horizontal_flip=True # Boolean: randomly flip inputs horizontally
)

# Number of augmented images
N = 10

img_dir=r'origin path'
img_path_list = ['vase.jpg', 'vase2.jpg', 'flowerpot.jpg']
img1 = os.path.join(img_dir, img_path_list[0])
augmented_img_path_list = get_augmented_img_path_list(img1, gen, N)