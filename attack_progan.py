import os
import time

import numpy as np
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
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score

"""Currently assumes jpg_prob, blur_prob 0 or 1"""

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    set_random_seed()
    opt = TrainOptions().parse()
    opt = get_processing_model(opt)
    test_loader = data.create_dataloader(opt)

    os.makedirs(os.path.join(opt.results_dir, opt.name), exist_ok=True)
    loss_fn = nn.BCEWithLogitsLoss()
    model = utils.get_model(opt, device)
    model = model.to(device)
    model.eval()

    writer = SummaryWriter(os.path.join(opt.results_dir, opt.name))

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
    fgsm = PGD(model)
    for i, (X, y) in enumerate(test_loader):
        print(i)
        X, y = X.to(device), y.to(torch.float32).to(device)
        adv_X = fgsm(X, y)
        y_pred.extend(model(adv_X).sigmoid().flatten().tolist())
        y_true.extend(y.flatten().tolist())
        # fgsm = FGSM(model)
        # adv_X = fgsm(X, y)
        # y_pred_raw.extend(model(X).flatten().tolist())
        # y_pred.extend(model(adv_X).flatten().tolist())
        # y_true.extend(y.to(torch.float32).flatten().tolist())
        # TP = (np.array(y_pred_raw) > 0).sum().item()
        # adv_TP = (np.array(y_pred)[y_pred_raw == 1] > 0).sum().item()
        # TN = (np.array(y_pred_raw) < 0).sum().item()
        # adv_TN = (np.array(y_pred)[y_pred_raw == 0] < 0).sum().item()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    # acc_raw = accuracy_score(y_true, y_pred_raw > 0)
    ap = average_precision_score(y_true, y_pred)
    logger.info(f"acc:{acc}")
    # logger.info(f"acc_raw:{acc_raw}")
    logger.info(f"real_acc:{r_acc}")
    logger.info(f"fake_acc:{f_acc}")
