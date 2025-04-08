import os

from torch.utils.data import DataLoader

from attacks import PGD
from data.datasets import CustomDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils
import data
from options import TrainOptions
from data.process import get_processing_model
from utils import set_random_seed
import numpy as np

if __name__ == '__main__':

    set_random_seed()
    opt = TrainOptions().parse()
    opt = get_processing_model(opt)
    test_dataset = CustomDataset(opt, 'test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    os.makedirs(os.path.join(opt.results_dir, opt.name), exist_ok=True)

    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))
    model = utils.get_model_tsne(opt)
    model_raw = utils.get_model(opt)
    # print(model)
    #
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    model_raw = model_raw.to(device)
    model.eval()
    model_raw.eval()

    features_pred, y_true = [], []
    for i, (X, y) in enumerate(test_loader):
        print(i)
        if i > 90:
            break
        X, y = X.to(device), y.to(device)
        real, fake = X[y == 0], X[y == 1]
        real_adv_label = torch.full((real.size(0),), 2).to(device)
        fake_adv_label = torch.full((fake.size(0),), 3).to(device)
        attack = PGD(model_raw)
        real_adv = attack(real, y[y == 0])
        fake_adv = attack(fake, y[y == 1])
        X = torch.cat((X, real_adv, fake_adv), dim=0)
        y = torch.cat((y, real_adv_label, fake_adv_label), dim=0)
        features = model(X)
        features = features.reshape(features.size(0), -1)
        # features = X.reshape(X.size(0), -1)
        features_pred.extend(features.tolist())
        y_true.extend(y.tolist())

    features_pred, y_true = np.array(features_pred), np.array(y_true)
    tsne = TSNE(n_components=2)
    tsne_features = tsne.fit_transform(features_pred)
    # 提取转换后的数据
    real_tsne = tsne_features[y_true == 0]
    fake_tsne = tsne_features[y_true == 1]
    real_adv_tsne = tsne_features[y_true == 2]
    fake_adv_tsne = tsne_features[y_true == 3]

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], label='sdv-real', alpha=0.5, color='tomato')
    plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1], label='sdv-fake', alpha=0.5, color='deepskyblue')
    plt.scatter(real_adv_tsne[:, 0], real_adv_tsne[:, 1], label='deepfake-fake', alpha=0.5, color='mediumpurple')
    plt.scatter(fake_adv_tsne[:, 0], fake_adv_tsne[:, 1], label='deepfake-real', alpha=0.5, color='yellowgreen')
    plt.title('deepfake&sdv', fontsize=18)
    plt.axis('off')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    # # 3D可视化展示
    # colors = ['g' if label == 0 else 'b' for label in y_true]
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(projection='2d')
    # ax.set_title('t-SNE process')
    # ax.scatter(tsne_features[:, 0], tsne_features[:, 1], c=colors, s=10)

    plt.savefig(os.path.join(opt.results_dir, opt.name, 'sdv.png'))
