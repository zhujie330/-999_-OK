import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from random import random, choice
import copy
from scipy import fftpack
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as TF


def data_augment(img):
    img = np.array(img)

    if random() < 0.5:
        sig = sample_continuous([0,3])
        gaussian_blur(img, sig)

    if random() < 0.5:
        method = sample_discrete(['cv2'])
        qual = sample_discrete([30,100])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def get_processing_model(opt):
    opt.dct_mean = torch.load('./weights/dct_mean_ff++.pth').cuda()
    opt.dct_var = torch.load('./weights/dct_var_ff++.pth').cuda()
    return opt


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


def custom_resize(img, opt):
    # width, height = img.size
    # print('before resize: '+str(width)+str(height))
    # quit()
    interp = sample_discrete(opt.rz_interp)
    # img = torchvision.transforms.Resize((int(height/2),int(width/2)))(img)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
    # return img


def dct2_wrapper(image, mean, var, log=True, epsilon=1e-12):
    """apply 2d-DCT to image of shape (H, W, C) uint8
    """
    # dct
    image = np.array(image)
    image = fftpack.dct(image, type=2, norm="ortho", axis=2)
    image = fftpack.dct(image, type=2, norm="ortho", axis=3)

    # log scale
    if log:
        image = np.abs(image)
        image += epsilon  # no zero in log
        image = np.log(image)
    # normalize
    image = (image - mean) / np.sqrt(var)
    return image


def idct2_wrapper(image, mean, var, log=True, epsilon=1e-12):
    image = np.array(image)
    # normalize
    image = image * np.sqrt(var) + mean
    # log scale
    if log:
        image -= epsilon  # no zero in log
        image = np.exp(image)

    image = fftpack.idct(image, type=2, norm="ortho", axis=2)
    image = fftpack.idct(image, type=2, norm="ortho", axis=3)
    return image


def processing(img, opt):
    input_img = copy.deepcopy(img)
    img = transforms.ToTensor()(input_img)
    #input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

    img = transforms.Resize(opt.loadSize, antialias=True)(img)
    img = transforms.CenterCrop(opt.CropSize)(img)
    return img
