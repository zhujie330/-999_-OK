import os

import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from data.datasets import CustomDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import time
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import utils
import data
import torch_dct as dct
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
    train_dataset = CustomDataset(opt, 'train')
    valid_dataset = CustomDataset(opt, 'valid')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    print('#training images = %d' % len(train_loader))
    print('#validation images = %d' % len(val_loader))
    os.makedirs(os.path.join(opt.results_dir, opt.name), exist_ok=True)
    train_writer = SummaryWriter(os.path.join(opt.results_dir, opt.name))

    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    model = utils.get_model(opt).to(device)
    # model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

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

    # Training
    prev_acc = 0
    total_steps = 0
    for epoch in range(opt.epochs):
        start_epoch_time = time.time()
        train_acc, val_acc = 0, 0
        model.train()
        for i, (X, y) in enumerate(train_loader):
            total_steps += 1
            X, y = X.to(device), y.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            # print((output.squeeze() > 0) * (y == 1))
            # train_acc = (((output.squeeze().sigmoid()[y == 0] < 0.5).sum().item()
            #               + (output.squeeze().sigmoid()[y == 1] > 0.5).sum().item())) / y.size(0)
            train_acc = (output.argmax(dim=1) == y).sum().item() / y.size(0)
            logger.info(f"epoch:{epoch}, loss: {loss.item()}")
            logger.info(f"epoch:{epoch}, train_acc:{train_acc}")

            train_writer.add_scalar('train/loss', loss, total_steps)
            train_writer.add_scalar('train/acc', train_acc, total_steps)

        model.eval()
        y_true, y_pred = [], []
        for j, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)

            output = model(X)
            y_pred.extend(output.argmax(dim=1).tolist())
            y_true.extend(y.tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
        f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
        acc = accuracy_score(y_true, y_pred)

        logger.info(f"acc:{acc}")
        logger.info(f"real_acc:{r_acc}")
        logger.info(f"fake_acc:{f_acc}")

        train_writer.add_scalar('val/acc', acc, epoch)
        train_writer.add_scalar('val/real_acc', r_acc, epoch)
        train_writer.add_scalar('val/fake_acc', f_acc, epoch)

        train_time = time.time()
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps
        }
        torch.save(state_dict, os.path.join(opt.results_dir, opt.name, 'CNNSpot.pth'))
