import os
import face_recognition
import streamlit as st
import torch
import psutil
from PIL import Image
import cv2
import numpy as np
import time
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
import tempfile
import time
import os
import logging
import torch.nn.functional as F
import requests
from io import BytesIO
logging.basicConfig(level=logging.DEBUG)
from modelscope import snapshot_download
print("哈哈哈")

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

device = torch.device('cpu')
st.set_page_config(page_title="Deepfake Detection", page_icon="🔎")


model_dir = os.path.join(tempfile.gettempdir(), 'model_use414')


model_file_path = os.path.join(model_dir, 'model1.pth')  # 假设模型文件是 model1.pth


if os.path.exists(model_file_path):
    st.write("✔️ 模型已加载")
else:
    st.write("⚠️ 由于 Git LFS 流量已达上线，自动转从 ModelScope 联网加载模型，请稍后")

    model_dir = snapshot_download('zhujie67o/model_use414')  # 通过ModelScope下载模型
    st.write("✔️ 模型已加载")

print(f"Using device: {device}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



def detect_faces(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image, faces


def preprocess(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # image = cv2.imread(img)
    # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(img)
    top, right, bottom, left = faces[0]
    face_img = img[top:bottom, left:right, :]
    face_img = torch.tensor(face_img / 255.0).permute(2, 0, 1)
    image_tensor = transform(face_img).unsqueeze(0)
    return image_tensor


# def predict_img(image_tensor):
#     logit = model(image_tensor.to(device))
#     cls = torch.argmax(logit, dim=1).item()
#     confidence = torch.softmax(logit, dim=1)[0][cls]
#     prediction = "real" if cls == 0 else "fake"
#     return prediction, confidence
def predict_img(image_tensor):
    print("开始预测")
    try:
        model.eval()
        with torch.no_grad():
            print("将图像张量移动到设备")
            image_tensor = image_tensor.to(device)
            print(f"图像张量大小: {image_tensor.size()}, 数据类型: {image_tensor.dtype}")
            print("模型推理开始")
            logit = model(image_tensor)
            print("模型推理结束")
            cls = torch.argmax(logit, dim=1).item()
            confidence = torch.softmax(logit, dim=1)[0][cls]
            prediction = "real" if cls == 0 else "fake"
            print("预测成功")
            return prediction, confidence.item()
    except Exception as e:
        raise RuntimeError(f"预测出错: {e}")


class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=None)

        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))


def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    # cv2.imwrite('./2.png',image*255)
    return image


def frame_extract(video):
    vidObj = cv2.VideoCapture(video)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


class validation_dataset():
    def __init__(self, video, sequence_length=100):
        self.video = video
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.count = sequence_length

    def get_dataset(self):
        frames = []
        word = st.empty()
        word.text('⌛extract frame...')
        bar = st.progress(0)

        for i, frame in enumerate(frame_extract(self.video)):
            # if(i % a == first_frame):
            bar.progress(i + 1)
            time.sleep(0.1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if (len(frames) == self.count):
                break
        # print("no of frames",len(frames))
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)


import torch.nn.functional as F
import streamlit as st
import logging

# 设置日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def predict(model, img):
    try:
        st.write("🟢 进入 predict 函数")
        logger.info("🟢 进入 predict 函数")

        st.write("📥 输入类型:", type(img))
        logger.info(f"📥 输入类型: {type(img)}")

        if isinstance(img, torch.Tensor):
            st.write("📐 输入 shape:", img.shape)
            logger.info(f"📐 输入 shape: {img.shape}")
        else:
            st.warning("⚠️ 输入 img 不是 torch.Tensor！")
            logger.warning("⚠️ 输入 img 不是 torch.Tensor！")

        img = img.to(device)
        st.write("✅ img.to(device) 成功")
        logger.info("✅ img.to(device) 成功")

        # 前向传播
        st.write("🚀 正在执行 model(img)")
        logger.info("🚀 正在执行 model(img)")

        output = model(img)
        st.write("✅ 前向传播完成，返回类型:", type(output))
        logger.info(f"✅ 前向传播完成，返回类型: {type(output)}")

        # 防止模型只返回一个结果时出错
        if isinstance(output, tuple):
            fmap, logits = output
            st.write("📦 fmap shape:", fmap.shape)
            st.write("📦 logits shape:", logits.shape)
            logger.info(f"📦 fmap shape: {fmap.shape}")
            logger.info(f"📦 logits shape: {logits.shape}")
        else:
            fmap = None
            logits = output
            st.warning("⚠️ 模型只返回了一个值，假设是 logits")
            logger.warning("⚠️ 模型只返回了一个值，假设是 logits")

        # 权重获取
        try:
            weight_softmax = model.linear1.weight.detach().cpu().numpy()
            st.write("🎯 获取 linear1 权重成功，shape:", weight_softmax.shape)
            logger.info(f"🎯 获取 linear1 权重成功，shape: {weight_softmax.shape}")
        except Exception as e:
            st.warning(f"⚠️ 获取 linear1 权重失败: {e}")
            logger.warning(f"⚠️ 获取 linear1 权重失败: {e}")

        logits = F.softmax(logits, dim=1)
        st.write("✅ Softmax 计算完成")
        logger.info("✅ Softmax 计算完成")

        _, prediction = torch.max(logits, 1)
        st.write("📊 预测结果标签:", int(prediction.item()))
        logger.info(f"📊 预测结果标签: {int(prediction.item())}")

        confidence = logits[:, int(prediction.item())].item() * 100
        st.write("📈 预测置信度:", confidence)
        logger.info(f"📈 预测置信度: {confidence}")

        return int(prediction.item()), confidence

    except Exception as e:
        st.error(f"❌ 模型推理出错: {e}")
        logger.error(f"❌ 模型推理出错: {e}")
        import traceback
        st.text(traceback.format_exc())
        logger.error(f"异常详情:\n{traceback.format_exc()}")
        raise RuntimeError(f"模型推理出错: {e}")



st.sidebar.header("🔎Deepfake Detection")

st.write("# Demo for Deepfake Detection🔎")
choice = st.sidebar.radio(label="What do you want to detect?", options=('Image'), index=0)

# 上传图片或视频
if choice == 'Image':
    uploaded_file = st.file_uploader(label="**choose the image you want to judge**", type=['jpg', 'png', 'jpeg'])
else:
    uploaded_file = st.file_uploader(label="**choose the video you want to judge**", type=['mp4', 'avi'])

# add_selectbox = st.sidebar.selectbox(
#     label="How would you like to be contacted?",
#     options=("Email", "Home phone", "Mobile phone"),
#     key="t1"
# )

# 显示结果
if uploaded_file is not None:
    if choice == 'Image':
        # 读取上传的图片
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='uploaded image')
        # model
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(2048, 2)
        # print("这里1")
        device = torch.device('cpu')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # states = torch.load(
        #      os.path.join("D:\\其他\\wehchatfile\\WeChat Files\\wxid_3hhhdkir3jfj22\\FileStorage\\File\\2024-07",
        #                   "CNNSpot.pth"))

        # states = torch.load("./model1.pth", map_location=torch.device("cpu"))

        states = torch.load(f"{model_dir}/model1.pth", map_location=torch.device("cpu"))
        # print("这里2")
        states = states['model']
        states = {key[2:]: value for key, value in states.items()}
        model.load_state_dict(states)
        model = model.to(device)
        model.eval()

        img_array = np.array(img)
        image_tensor = preprocess(img_array)
        # 检测人脸按钮
        if st.button('**start to detect**'):
            print("哈哈1")
            try:
                print(f"Memory usage before prediction: {psutil.virtual_memory().percent}%")
                prediction, confidence = predict_img(image_tensor)
                print(f"Memory usage after prediction: {psutil.virtual_memory().percent}%")
            except Exception as e:
                print(f"Error during prediction: {e}")
                st.error(f"Error during prediction: {e}")
            else:
                print("哈哈2")
                st.info(f"📋the face in image is **{prediction}**")
                print("哈哈3")
                st.info(f"📋the confidence is **{confidence}**")
                print("哈哈4")



