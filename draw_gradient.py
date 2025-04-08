import os

import cv2
from PIL import Image
from saliency.smoothgrad import SmoothGrad
from torchvision import transforms

from attacks import PGD

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils
from options import TrainOptions
from data.process import get_processing_model
from utils import set_random_seed
import numpy as np


def compute_gradient(input_image, model):
    model.eval()
    input_image.requires_grad = True
    output = model(input_image)
    pred_class = torch.argmax(output, dim=1)
    loss = output[0, pred_class[0]]
    loss.backward()
    gradient = input_image.grad.data
    return gradient


def visualize_heatmap(gradient):
    gradient = gradient.squeeze().cpu().numpy()
    gradient = np.amax(np.abs(gradient), axis=0)
    gradient /= np.max(gradient)  # [0, 1]
    # OpenCV 将梯度显著图转换成热度图
    heatmap = cv2.applyColorMap(np.uint8(255 * gradient), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    return heatmap, gradient


if __name__ == '__main__':
    set_random_seed()
    opt = TrainOptions().parse()
    # opt = get_processing_model(opt)

    os.makedirs(os.path.join(opt.results_dir, opt.name), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    model = utils.get_model(opt)
    # print(model)
    #
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([transforms.Resize(opt.loadSize, antialias=True),
                                    transforms.CenterCrop(opt.CropSize),
                                    transforms.ToTensor()])

    # 读取并预处理图像
    image_path = '/home/zhangruixuan/dataset/FF++/exp/youtube/41001.png'
    # image_path = '/home/zhangruixuan/FF++_fake/0.png'
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批处理维度
    image = image.to(device)
    # attack = PGD(model)
    # image = attack(image, torch.tensor([1]))
    # perturb = adv_image - image
    # perturb = perturb.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    # perturb /= np.max(perturb)
    # plt.imshow(perturb)
    # plt.show()

    input_image = image  # 假设输入图像大小为 224x224
    # print((input_image > 1.).sum().item())
    # print((input_image < 0.).sum().item())

    # 计算梯度
    gradient = compute_gradient(input_image, model)

    # 可视化热度图
    heatmap, saliency = visualize_heatmap(gradient)

    # 将输入图像转换成 numpy 数组
    input_image_np = input_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    # 将热度图和输入图像相加
    superimposed_img = heatmap + input_image_np

    # 归一化相加后的图像
    superimposed_img /= np.max(superimposed_img)

    # 显示相加后的图像
    # plt.imshow(superimposed_img)
    # plt.axis('off')
    # plt.show()
    # plt.savefig(os.path.join(opt.results_dir, opt.name, 'neuraltextures_adv.png'))

    plt.subplot(1, 3, 1)  # 1行3列的第二个
    plt.title("ad_img")
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(np.float32(input_image_np))

    # 第二个子图
    plt.subplot(1, 3, 2)  # 1行3列的第二个
    plt.title("saliency_map")
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(saliency)

    # 第三个子图
    plt.subplot(1, 3, 3)  # 1行3列的第三个
    plt.title("saliency_map_with_img")
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(superimposed_img)

    plt.show()
    plt.savefig(os.path.join(opt.results_dir, opt.name, 'neuraltextures_ad.png'))
