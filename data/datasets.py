from random import sample
from PIL import ImageFile
import os
from .process import *
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def loadpathslist(root, flag):
    result = []
    classes = os.listdir(root)
    if flag == 0:
        dirpaths = [cls for cls in classes if cls == 'youtube']
    else:
        dirpaths = [cls for cls in classes if cls != 'youtube']
    for path in dirpaths:
        imgpaths = os.listdir(root + '/' + path)
        for imgpath in imgpaths:
            result.append(root + '/' + path + '/' + imgpath)
    return result

# def loadpathslist(root, flag):
#     result = []
#     classes = os.listdir(root)
#
#     if flag == 0:
#         dirpaths = [cls for cls in classes if cls == '0_real']
#     else:
#         dirpaths = [cls for cls in classes if cls != '0_real']
#     for path in dirpaths:
#         imgpaths = os.listdir(root + '/' + path)
#         for imgpath in imgpaths:
#             result.append(root + '/' + path + '/' + imgpath)
#     return result


# def loadpathslist(root, name):
#     classes = os.listdir(root)
#     result = []
#     for cls in classes:
#         if cls == name:
#             paths = os.listdir(root + '/' + cls)
#             for path in paths:
#                 result.append(root + '/' + cls + '/' + path)
#             return result


class CustomDataset(Dataset):
    def __init__(self, opt, mode):
        self.opt = opt
        self.root = os.path.join(opt.dataroot, mode)
        self.transform = transforms.Compose([transforms.Resize(opt.loadSize, antialias=True),
                                             transforms.CenterCrop(opt.CropSize),
                                             transforms.ToTensor()])

        real_img_list = loadpathslist(self.root, 0)
        real_label_list = [int(0) for _ in range(len(real_img_list))]
        # fake
        # fake_img_list_df = loadpathslist(self.root, 'Deepfakes')
        # fake_img_list_f2 = loadpathslist(self.root, 'Face2Face')
        # fake_img_list_fs = loadpathslist(self.root, 'FaceShifter')
        # fake_img_list_fsw = loadpathslist(self.root, 'FaceSwap')
        # fake_img_list_nt = loadpathslist(self.root, 'NeuralTextures')
        fake_img_list = loadpathslist(self.root, 1)
        fake_label_list = [int(1) for _ in range(len(fake_img_list))]

        # real_img_list = loadpathslist(self.root, '0_real')
        # real_label_list = [0 for _ in range(len(real_img_list))]
        # fake_img_list = loadpathslist(self.root, '1_fake')
        # fake_label_list = [1 for _ in range(len(fake_img_list))]
        self.img = real_img_list + fake_img_list
        self.label = real_label_list + fake_label_list

        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))

    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.label)
