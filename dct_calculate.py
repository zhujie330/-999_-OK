import torch_dct as dct

import torch
from PIL import Image
from torchvision import transforms

import utils
import data
from attacks import *
from options import TrainOptions
from data.process import get_processing_model
from utils import set_random_seed
from data.datasets import read_data_new
import utils
print("这里运行了777")
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    set_random_seed()
    opt = TrainOptions().parse()
    # test_loader = data.create_dataloader(opt)
    dataset = read_data_new(opt)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((opt.loadSize)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    transforms.CenterCrop(opt.CropSize)])

    images = []
    for path in dataset.img:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        dct_img = dct.dct(img_tensor)
        images.append(dct_img)
    mean, var = utils.welford(images)
    torch.save(mean, '/home/zhangruixuan/code/fast_adversarial/weights/dct_mean_ff++.pth')
    torch.save(var, '/home/zhangruixuan/code/fast_adversarial/weights/dct_var_ff++.pth')
