import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.models import ResNet50_Weights
import torch_dct as dct
from .resnet import resnet50


# def dct(x, norm=None):
#     """
#     Discrete Cosine Transform, Type II (a.k.a. the DCT)
#
#     For the meaning of the parameter `norm`, see:
#     https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
#
#     :param x: the input signal
#     :param norm: the normalization, None or 'ortho'
#     :return: the DCT-II of the signal over the last dimension
#     """
#     x_shape = x.shape
#     N = x_shape[-1]
#     x = x.contiguous().view(-1, N)
#
#     v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
#
#     Vc = torch.fft.fft(v)
#
#     k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
#     W_r = torch.cos(k)
#     W_i = torch.sin(k)
#
#     # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
#     V = Vc.real * W_r - Vc.imag * W_i
#     if norm == 'ortho':
#         V[:, 0] /= np.sqrt(N) * 2
#         V[:, 1:] /= np.sqrt(N / 2) * 2
#
#     V = 2 * V.view(*x_shape)
#
#     return V
#
# def dct_2d(x, norm=None):
#     """
#     2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
#
#     For the meaning of the parameter `norm`, see:
#     https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
#
#     :param x: the input signal
#     :param norm: the normalization, None or 'ortho'
#     :return: the DCT-II of the signal over the last 2 dimensions
#     """
#     X1 = dct(x, norm=norm)
#     X2 = dct(X1.transpose(-1, -2), norm=norm)
#     return X2.transpose(-1, -2)

class DCTFunction(nn.Module):
    def __init__(self, dct_mean, dct_var):
        super(DCTFunction, self).__init__()
        self.log = True
        self.epsilon = 1e-12
        self.dct_mean = dct_mean.unsqueeze(0).float()
        self.dct_var = dct_var.unsqueeze(0).float()

    def forward(self, x):
        batch_size = x.size(0)
        dct_mean = self.dct_mean
        dct_var = self.dct_var
        # DCT
        # x = x * 255.0 # [0, 1.0] to [0, 255.0]
        x = dct.dct_2d(x, norm='ortho')
        if self.log:
            x = torch.abs(x)
            x += self.epsilon  # no zero in log
            x = torch.log(x)
        # normalize  BCHW
        x = (x - dct_mean) / torch.sqrt(dct_var)
        return x


def get_DCTA(opt):
    dct_layer = DCTFunction(dct_mean=opt.dct_mean, dct_var=opt.dct_var)
    # Freq_basemodel = resnet50(pretrained=True)

    if opt.task == 'train':
        Freq_basemodel = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        Freq_basemodel.fc = nn.Linear(2048, 1)
        torch.nn.init.normal_(Freq_basemodel.fc.weight.data, 0.0, 0.02)
    else:
        Freq_basemodel = torchvision.models.resnet50(num_classes=1)

    return nn.Sequential(dct_layer, Freq_basemodel)
