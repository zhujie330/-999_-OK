import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from attacks import PGD

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# 加载ResNet50模型
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(2048, 2)
states = torch.load(
    os.path.join("/home/zhangruixuan/code/fast_adversarial/results/CNNSpot", "pgd_train", "CNNSpot.pth"))
states = states['model']
states = {key[2:]: value for key, value in states.items()}
model.load_state_dict(states)
# print(model)

# layer_name = 'layer4'
# layer = None
# # 递归搜索子模块
# for name, module in model.named_children():
#     if name == layer_name:
#         layer = module
#         break
#
# if layer is None:
#     raise ValueError(f"Layer with name '{layer_name}' not found in the model.")

# model_new = torch.nn.Sequential(*(list(model.children())[:list(model.children()).index(layer)]))
model_new = torch.nn.Sequential(*(list(model.children())[:-2]))
model_new = model_new.to(device)
model.to(device)

# 加载并预处理图像
img_path = '/home/zhangruixuan/dataset/FF++/exp/youtube/41009.png'
img = Image.open(img_path)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)  # 添加一个维度，因为模型输入需要一个batch
img_tensor = img_tensor.to(device)
attack = PGD(model)
img_tensor = attack(img_tensor, torch.tensor([1]))

# 获取特征图
model.eval()
with torch.no_grad():
    feature_maps = model_new(img_tensor.to(device))

# 可视化特征图
num_feature_maps = feature_maps.shape[1]
square = 16
# plt.figure(figsize=(12, 12))
for i in range(256):
    plt.subplot(square, square, i + 1)
    plt.imshow(feature_maps[0, i].cpu(), cmap='viridis')
    plt.axis('off')
# plt.imshow(feature_maps[0].view(32, -1).cpu(), cmap='viridis')
plt.show()
# plt.savefig('feature_map.png')
