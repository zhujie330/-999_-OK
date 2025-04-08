from random import sample
from PIL import ImageFile
import os
import shutil
from data.process import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
print("这里运行了999")

def loadpathslist(root, name):
    classes = os.listdir(root)
    result = []
    for cls in classes:
        if cls == name:
            dirpaths = os.listdir(root + '/' + cls)
            for path in dirpaths:
                imgpaths = os.listdir(root + '/' + cls + '/' + path)
                for imgpath in imgpaths:
                    result.append(root + '/' + cls + '/' + path + '/' + imgpath)
            if cls == 'youtube':
                return sample(result, 50000)
            return sample(result, 10000)


# 遍历路径列表
def copy_images_by_paths(destination_folder, path_list, start, end):
    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)

    # 遍历路径列表
    for i in range(start, end):
        # 检查源文件是否存在
        if os.path.exists(path_list[i]):
            # 构建目标文件路径
            destination_path = os.path.join(destination_folder, f'{i}.png')
            # 复制文件到目标文件夹
            shutil.copyfile(path_list[i], destination_path)


def dataset_split(mode):
    root = '/home/zhangruixuan/dataset/FF++C32'
    real_img_list = loadpathslist(root, 'youtube')

    # fake
    fake_img_list_df = loadpathslist(root, 'Deepfakes')
    fake_img_list_f2 = loadpathslist(root, 'Face2Face')
    fake_img_list_fs = loadpathslist(root, 'FaceShifter')
    fake_img_list_fsw = loadpathslist(root, 'FaceSwap')
    fake_img_list_nt = loadpathslist(root, 'NeuralTextures')

    start_real, end_real, start_fake, end_fake = 0, 0, 0, 0
    destination_folder=''
    if mode == 'train':
        start_real, end_real, start_fake, end_fake = 0, 36000, 0, 7200
        destination_folder='/home/zhangruixuan/dataset/FF++/train'
    elif mode == 'valid':
        start_real, end_real, start_fake, end_fake = 36000, 41000, 7200, 8200
        destination_folder = '/home/zhangruixuan/dataset/FF++/valid'
    elif mode == 'exp':
        start_real, end_real, start_fake, end_fake = 41000, 43000, 8200, 8600
        destination_folder = '/home/zhangruixuan/dataset/FF++/exp'
    elif mode == 'test':
        start_real, end_real, start_fake, end_fake = 43000, 50000, 8600, 10000
        destination_folder = '/home/zhangruixuan/dataset/FF++/test'

    copy_images_by_paths(os.path.join(destination_folder,"youtube"),real_img_list,start_real,end_real)
    copy_images_by_paths(os.path.join(destination_folder,"deepfakes"),fake_img_list_df,start_fake,end_fake)
    copy_images_by_paths(os.path.join(destination_folder,"face2face"), fake_img_list_f2, start_fake, end_fake)
    copy_images_by_paths(os.path.join(destination_folder,"faceshifter"), fake_img_list_fs, start_fake, end_fake)
    copy_images_by_paths(os.path.join(destination_folder,"faceswap"), fake_img_list_fsw, start_fake, end_fake)
    copy_images_by_paths(os.path.join(destination_folder,"neuraltextures"), fake_img_list_nt, start_fake, end_fake)


if __name__ == '__main__':
    dataset_split('train')
    dataset_split('valid')
    dataset_split('exp')
    dataset_split('test')
