import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
# 定义图像文件夹路径
print("这里运行了7777")
image_folder = '/home/zhangruixuan/dataset/CNNsyth/cyclegan'
# os.listdir(image_folder)
# image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)
#                if file.endswith(('.png', '.jpg', '.jpeg'))]

image_folders = os.listdir(image_folder)

image_files = []
# 获取图像文件夹中所有图像文件的文件名
for folder in image_folders:
    folder_path = os.path.join(image_folder, folder)
    files = [os.path.join(folder_path, '1_fake',file) for file in os.listdir(os.path.join(folder_path, '1_fake')) if
             file.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.extend(files)

# 创建一个空列表用于存储灰度图像数据
gray_images = []

# 遍历图像文件列表，读入并转换成灰度图像
for i, image_file in enumerate(image_files):
    image = plt.imread(image_file)
    gray_image = np.mean(image, axis=2)
    gray_image = cv2.resize(gray_image, (256, 256))
    # 将灰度图像添加到列表中
    gray_images.append(gray_image)

# 计算平均灰度图像
avg_gray_image = np.mean(gray_images, axis=0)

# 计算灰度图的傅里叶变换
# frequencies = np.fft.fft2(avg_gray_image)
# frequencies_shifted = np.fft.fftshift(frequencies)
# magnitude_spectrum = np.log(np.abs(frequencies_shifted) + 1)  # 取对数，增强可视化效果
frequencies = fftpack.dctn(avg_gray_image, norm='ortho')
magnitude_spectrum = np.log(np.abs(frequencies + 1e-12))
# 绘制彩色频谱图
plt.imshow(magnitude_spectrum, cmap='plasma')  # 使用彩色的色谱，比如'plasma'
plt.colorbar()  # 添加颜色条
plt.title('Magnitude Spectrum-dct_cycleggan')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.show()
plt.savefig('/home/zhangruixuan/spectrum/dct_cyclegan.png')
