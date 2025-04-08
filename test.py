import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import utils
import data
from attacks import PGD
from data.datasets import CustomDataset
from options import TrainOptions
from data.process import get_processing_model
from utils import set_random_seed
import logging

"""Currently assumes jpg_prob, blur_prob 0 or 1"""

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    set_random_seed()
    opt = TrainOptions().parse()
    # opt = get_processing_model(opt)
    test_dataset = CustomDataset(opt, 'test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
    print('#training images = %d' % len(test_loader))

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
        filename=os.path.join(opt.results_dir, opt.name, 'exp3.log'))
    logger = logging.getLogger(__name__)
    # 创建一个控制台处理器并添加到日志记录器中
    console_handler = logging.StreamHandler()  # 创建一个控制台处理器
    console_handler.setLevel(logging.INFO)  # 设置处理器级别为 INFO
    formatter = logging.Formatter('[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')  # 创建一个格式化器
    console_handler.setFormatter(formatter)  # 将格式化器添加到处理器中
    logger.addHandler(console_handler)  # 将处理器添加到日志记录器中

    y_true, y_pred, y_pred_ad = [], [], []
    for j, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        attack = PGD(model)
        X_ad = attack(X, y)
        output = model(X)
        y_pred.extend(output.argmax(dim=1).tolist())
        y_true.extend(y.tolist())
        output = model(X_ad)
        y_pred_ad.extend(output.argmax(dim=1).tolist())

    y_true, y_pred, y_pred_ad = np.array(y_true), np.array(y_pred), np.array(y_pred_ad)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
    r_acc_ad = accuracy_score(y_true[y_true == 0], y_pred_ad[y_true == 0])
    f_acc_ad = accuracy_score(y_true[y_true == 1], y_pred_ad[y_true == 1])
    # acc = accuracy_score(y_true, y_pred)

    # logger.info(f"acc:{acc}")
    logger.info(f"real_acc:{r_acc}")
    logger.info(f"fake_acc:{f_acc}")
    logger.info(f"real_acc_ad:{r_acc_ad}")
    logger.info(f"fake_acc_ad:{f_acc_ad}")


