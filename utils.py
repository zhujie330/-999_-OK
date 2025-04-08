import os
from collections import OrderedDict
import torch_dct as dct
import numpy
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms
from model import dcta
from model.dcta import DCTFunction
# from torchvision.models import ResNet50_Weights


def get_limit(opt):
    upper_limit = ((numpy.array(1) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])  # 处理后的两端极限值
    upper_limit = torch.tensor(upper_limit).view(1, 3, 1, 1)
    upper_limit = upper_limit.expand(1, 3, 224, 224)
    lower_limit = ((numpy.array(0) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])  # 处理后的两端极限值
    lower_limit = torch.tensor(lower_limit).view(1, 3, 1, 1)
    lower_limit = lower_limit.expand(1, 3, 224, 224)
    return upper_limit, lower_limit


def get_epsilon(opt):
    eps_max = ((numpy.array(opt.epsilon / 255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])  # 处理后的两端极限值
    eps_min = ((numpy.array(-opt.epsilon / 255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
    return torch.tensor(eps_max), torch.tensor(eps_min)


def dct_fn(image, opt, log=True, epsilon=1e-12):
    image = dct.dct_2d(image, norm='ortho').to(torch.float32)
    # log scale
    if log:
        image = torch.abs(image)
        image = torch.log(image + epsilon)
    # normalize
    dct_mean = torch.tensor(opt.dct_mean)
    dct_var = torch.tensor(np.sqrt(opt.dct_var))
    image = (image - dct_mean) / dct_var
    return image


def attack_fgsm(opt, X, y, model, loss_fn, device):
    delta = torch.zeros_like(X).to(torch.float32)
    X.requires_grad = True
    output = model(X.to(device))
    loss = loss_fn(output.squeeze(), y.to(torch.float32).to(device))
    loss.backward()
    grad = X.grad.detach()
    with torch.no_grad():
        delta += opt.alpha * torch.sign(grad)
        delta.clamp_(-opt.alpha / 255, opt.alpha / 255)
        delta.clamp_(- X, 1 - X)
    return delta


def attack_pgd(opt, X, y, model, loss_fn, device):
    delta = torch.zeros_like(X)
    delta.uniform_(-opt.epsilon / 255, opt.epsilon / 255)
    delta.clamp_(- X, 1 - X)
    for j in range(opt.pgd_step):
        delta.requires_grad = True
        input_X = X + delta
        input_X = input_X.to(torch.float32).to(device)
        output = model(input_X)
        loss = loss_fn(output.squeeze(), y.to(torch.float32).to(device))
        loss.backward()
        grad = delta.grad
        with torch.no_grad():
            delta += opt.pgd_alpha * torch.sign(grad)
            delta.clamp_(-opt.epsilon / 255, opt.epsilon / 255)
            delta.clamp_(- X, 1 - X)
    return delta


def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def get_model(opt):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    classifier = torchvision.models.resnet50(pretrained=True)
    # classifier = torchvision.models.resnet50()
    classifier.fc = torch.nn.Linear(2048, 2)

    if opt.detect_method == 'CNNSpot':
        model = torch.nn.Sequential(normalize, classifier)
        if opt.task == 'train':
            torch.nn.init.normal_(classifier.fc.weight.data, 0.0, opt.init_gain)
        else:
            states = torch.load(os.path.join(opt.results_dir, "raw_train", "CNNSpot.pth"))
            # states=torch.load('/home/zhangruixuan/CNNSpot_gan.pth',device='cuda:0')
            model.load_state_dict(states['model'])
            # torch.nn.init.normal_(classifier.fc.weight.data, 0.0, opt.init_gain)
        return model
    else:
        dct_layer = DCTFunction(dct_mean=opt.dct_mean, dct_var=opt.dct_var)
        model = torch.nn.Sequential(normalize, dct_layer, classifier)
        if opt.task == 'train':
            torch.nn.init.normal_(classifier.fc.weight.data, 0.0, opt.init_gain)
        else:
            states = torch.load(os.path.join(opt.results_dir, "train_raw", "DCTA.pth"))
            model.load_state_dict(states['model'])
        return model


def get_model_tsne(opt):
    model = get_model(opt)
    return torch.nn.Sequential(model[0], *(list(model[1].children())[:-1]))


def _welford_update(existing_aggregate, new_value):
    count, mean, M2 = existing_aggregate
    if count is None:
        count, mean, M2 = 0, torch.zeros_like(new_value), torch.zeros_like(new_value)

    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2

    return (count, mean, M2)


def _welford_finalize(existing_aggregate):
    count, mean, M2 = existing_aggregate
    mean, variance, sample_variance = (mean, M2 / count, M2 / (count - 1))
    if count < 2:
        return (torch.tensor(float("nan")), torch.tensor(float("nan")), torch.tensor(float("nan")))
    else:
        return (mean, variance, sample_variance)


def welford(sample):
    existing_aggregate = (None, None, None)
    for data in sample:
        existing_aggregate = _welford_update(existing_aggregate, data)

    # sample variance only for calculation
    return _welford_finalize(existing_aggregate)[:-1]
