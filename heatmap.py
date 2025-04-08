import os
from torchcam.utils import overlay_mask
from PIL import Image

from torchvision import transforms

from attacks import PGD

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils
import data
from options import TrainOptions
from data.process import get_processing_model
from utils import set_random_seed
import numpy as np
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

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
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(opt.loadSize, antialias=True),
                                    transforms.CenterCrop(opt.CropSize)])

    # 读取并预处理图像
    image_path = '/home/zhangruixuan/FF++_real/0.png'
    # image_path = '/home/zhangruixuan/FF++_fake/0.png'
    # image = Image.open(image_path)
    # image = transform(image).unsqueeze(0)  # 添加批处理维度
    image = read_image(image_path)
    # Preprocess it for your chosen model
    in_tensor = image / 255.
    in_tensor=in_tensor.to(device)
    # image = image.to(device)
    # attack = PGD(model)
    # adv_image = attack(image, torch.tensor([1]))
    with SmoothGradCAMpp(model) as cam_extractor:
        # Preprocess your data and feed it to the model
        out = model(in_tensor.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(opt.results_dir, opt.name,'real_0.png'))
