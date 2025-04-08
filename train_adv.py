import os
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from attacks.fgsm_random import RFGSM
from data.datasets import CustomDataset
import torch.nn.functional as F
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import time
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import utils
import data
import torch_dct as dct

from attacks import FGSM, PGD, MIFGSM
from options import TrainOptions
from data.process import get_processing_model
from utils import set_random_seed
import logging

"""Currently assumes jpg_prob, blur_prob 0 or 1"""

if __name__ == '__main__':
    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    set_random_seed()
    opt = TrainOptions().parse()
    # opt = get_processing_model(opt)
    train_dataset = CustomDataset(opt, 'train')
    valid_dataset = CustomDataset(opt, 'valid')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True)
    print('#training images = %d' % len(train_loader))
    print('#validation images = %d' % len(val_loader))
    os.makedirs(os.path.join(opt.results_dir, opt.name), exist_ok=True)
    train_writer = SummaryWriter(os.path.join(opt.results_dir, opt.name))

    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    model = utils.get_model(opt)
    model = model.to(device)
    # model = nn.DataParallel(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr_max, momentum=opt.momentum,
    #                             weight_decay=opt.weight_decay)
    # if opt.delta_init == 'previous':
    #     delta = torch.zeros(opt.batch_size, 3, 224, 224)
    #
    # lr_steps = opt.epochs * len(train_loader)
    # if opt.lr_schedule == 'cyclic':
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=opt.lr_min, max_lr=opt.lr_max,
    #                                                   step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    # elif opt.lr_schedule == 'multistep':
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4],
    #                                                      gamma=0.1)

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

        for i, (X, y) in enumerate(train_loader):
            if i == 0:
                first_batch = X, y
            total_steps += 1

            # X = (X + torch.zeros_like(X).uniform_(-8 / 255, 8 / 255)).clamp(0, 1)

            X, y = X.to(device), y.to(device)
            # index = int(y.size(0) / 2)
            # X_adv = X[y == 1]
            # y_adv = y[y == 1]
            # real, fake = X[y == 0], X[y == 1]

            model.eval()
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
            # X_adv = attack(X_adv, y_adv)
            # real_adv_label = torch.full((real.size(0),), 2).to(device)
            # fake_adv_label = torch.full((fake.size(0),), 3).to(device)
            # real_adv = attack(real, y[y == 0])
            # fake_adv = attack(fake, y[y == 1])
            X_adv = attack(X, y)
            # X = attack(X, y)
            # X = torch.cat((X, X_adv), dim=0)
            # y = torch.cat((y, y_adv), dim=0)
            # X = torch.cat((X, real_adv, fake_adv), dim=0)
            # y = torch.cat((y, real_adv_label, fake_adv_label), dim=0)

            # update theta
            model.train()
            optimizer.zero_grad()
            output_adv = model(X_adv)
            output_clean = model(X)

            #clean loss
            loss_clean = loss_fn(output_clean, y)


            # robust loss
            criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
            loss_robust = criterion_kl(F.log_softmax(output_adv, dim=1),
                                       F.softmax(output_clean, dim=1))
            loss = loss_clean + 5.0 * loss_robust

            # output = model(X)
            # loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            # # update theta
            # model.train()
            # optimizer.zero_grad()
            # # output = model(adv_X)
            # output = model(X)
            # loss = loss_fn(output, y)
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            # print((output.squeeze() > 0) * (y == 1))
            # train_acc = (((output.squeeze().sigmoid()[y == 0] < 0.5).sum().item()
            #               + (output.squeeze().sigmoid()[y == 1] > 0.5).sum().item())) / y.size(0)
            train_acc = (output_adv.argmax(dim=1) == y).sum().item() / y.size(0)
            logger.info(f"epoch:{epoch}, loss: {loss.item()}")
            logger.info(f"epoch:{epoch}, train_acc:{train_acc}")

            train_writer.add_scalar('train/loss', loss, total_steps)
            train_writer.add_scalar('train/acc', train_acc, total_steps)

            # state_dict = {
            #     # 'model': model.module.state_dict(),
            #     'model': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'total_steps': total_steps
            # }
            # torch.save(state_dict, os.path.join(opt.results_dir, opt.name, 'CNNSpot_adv.pth'))

            # # early stop
            # pgd = PGD(model)
            # X_1, y_1 = first_batch
            # X_1, y_1 = X_1.to(device), y_1.to(device)
            # adv_X_1 = pgd(X_1, y_1.to(torch.float32))
            # output = model(adv_X_1)
            # robust_acc = (((output.squeeze().sigmoid()[y_1 == 0] < 0.5).sum().item()
            #                  +(output.squeeze().sigmoid()[y_1 == 1] > 0.5).sum().item()))/y_1.size(0)

        # if train_acc - prev_acc < -0.2:
        #     break
        # # prev_acc = robust_acc
        # prev_acc = train_acc

        # best_state_dict = model.state_dict()
        # logger.info(f"epoch:{epoch}, robust_acc: {robust_acc}")
        # train_writer.add_scalar('val/acc', robust_acc, epoch)

        model.eval()
        y_true, y_pred, y_pred_ad = [], [], []
        for j, (X, y) in enumerate(val_loader):
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

        # train_writer.add_scalar('val/acc', acc, epoch)
        train_writer.add_scalar('val/real_acc', r_acc, epoch)
        train_writer.add_scalar('val/fake_acc', f_acc, epoch)
        train_writer.add_scalar('val/real_ad_acc', r_acc_ad, epoch)
        train_writer.add_scalar('val/fake_ad_acc', f_acc_ad, epoch)

        train_time = time.time()
        state_dict = {
            # 'model': model.module.state_dict(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps
        }
        torch.save(state_dict, os.path.join(opt.results_dir, opt.name, 'CNNSpot.pth'))
