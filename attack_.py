import os

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader

from attacks.fgsm_random import RFGSM
from data.datasets import CustomDataset
import torchvision.utils as tvu
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as tvu
import utils
import data
from attacks import *
from options import TrainOptions
from data.process import get_processing_model
from utils import set_random_seed
import logging
from torchvision import transforms

"""Currently assumes jpg_prob, blur_prob 0 or 1"""

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    set_random_seed()
    opt = TrainOptions().parse()
    opt = get_processing_model(opt)
    test_dataset = CustomDataset(opt, 'exp')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    os.makedirs(os.path.join(opt.results_dir, opt.name), exist_ok=True)
    # loss_fn = nn.BCEWithLogitsLoss()
    model = utils.get_model(opt)
    # model = nn.DataParallel(model,device_ids=[1,0,2,3]).to(device)
    model = model.to(device)
    model.eval()

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(opt.results_dir, opt.name, 'output.log'))
    logger = logging.getLogger(__name__)
    # 创建一个控制台处理器并添加到日志记录器中
    console_handler = logging.StreamHandler()  # 创建一个控制台处理器
    console_handler.setLevel(logging.INFO)  # 设置处理器级别为 INFO
    formatter = logging.Formatter('[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')  # 创建一个格式化器
    console_handler.setFormatter(formatter)  # 将格式化器添加到处理器中
    logger.addHandler(console_handler)  # 将处理器添加到日志记录器中

    y_true, y_pred = [], []
    for i, (X, y) in enumerate(test_loader):
        print(i)
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            # preds = model(X).sigmoid().flatten()
            # condition = ((y == 0) & (preds < 0.5)) | ((y == 1) & (preds > 0.5))
            output = model(X)
            condition = (output.argmax(dim=1) == y)
            X = X[condition]
            y = y[condition]

        if X.size(0) > 0:
            if opt.adv_attack == 'fgsm':
                attack = FGSM(model)
            elif opt.adv_attack == 'pgd':
                attack = PGD(model)
            elif opt.adv_attack == 'ifgsm':
                attack = PGD(model, random_start=False)
            elif opt.adv_attack == 'mifgsm':
                attack = MIFGSM(model)
            elif opt.adv_attack == 'rfgsm':
                attack = RFGSM(model)
            adv_X = attack(X, y)
            # os.makedirs('/home/zhangruixuan/FF++_adv_real', exist_ok=True)
            # os.makedirs('/home/zhangruixuan/FF++_real', exist_ok=True)
            # os.makedirs('/home/zhangruixuan/FF++_adv_fake', exist_ok=True)
            # os.makedirs('/home/zhangruixuan/FF++_fake', exist_ok=True)
            # X_real, X_fake = X[y == 0], X[y == 1]
            # adv_X_real, adv_X_fake = adv_X[y == 0], adv_X[y == 1]

            # for i in range(len(X_real)):
            #     tvu.save_image(X_real[i], os.path.join('/home/zhangruixuan/FF++_real', f'{i}.png'))
            # for i in range(len(X_fake)):
            #     tvu.save_image(X_fake[i], os.path.join('/home/zhangruixuan/FF++_fake', f'{i}.png'))
            # for i in range(len(adv_X_real)):
            #     tvu.save_image(adv_X_real[i], os.path.join('/home/zhangruixuan/FF++_adv_real', f'{i}.png'))
            # for i in range(len(adv_X_fake)):
            #     tvu.save_image(adv_X_fake[i], os.path.join('/home/zhangruixuan/FF++_adv_fake', f'{i}.png'))
            # y_pred.extend(model(adv_X).sigmoid().flatten().tolist())
            y_pred.extend(model(adv_X).argmax(dim=1).tolist())
            y_true.extend(y.tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    # f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    # acc = accuracy_score(y_true, y_pred > 0.5)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
    acc = accuracy_score(y_true, y_pred)

    logger.info(f"acc:{acc}")
    # logger.info(f"acc_raw:{acc_raw}")
    logger.info(f"real_acc:{r_acc}")
    logger.info(f"fake_acc:{f_acc}")
