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
from utils_model import get_model_dir 

logging.basicConfig(level=logging.DEBUG)
from modelscope import snapshot_download
import base64

print("哈哈哈")
st.set_page_config(page_title="Deepfake Detection", page_icon="🔎")
st.sidebar.header("🔎Deepfake Detection")
st.write("# Demo for Deepfake Detection🔎")
st.write("⚠️ 由于 Git LFS 流量已达上线，自动转从 ModelScope 联网加载模型，请稍后")

device = torch.device('cpu')
model_dir = get_model_dir()
model_file_path = os.path.join(model_dir, 'model1.pth')

if os.path.exists(model_file_path):
    st.write("✔️ 模型已加载")
else:
     st.write("⚠️ 模型文件未找到，请稍候重试")

   
    st.write("✔️ 模型已加载, 接下来你可以选择使用系统为您准备的一些测试图片 或者 选择你本地想要上传的图片进行检测")

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


def predict_img(image_tensor):
    print("开始预测")
    try:
        st.session_state.model.eval()
        with torch.no_grad():
            print("将图像张量移动到设备")
            image_tensor = image_tensor.to(device)
            print(f"图像张量大小: {image_tensor.size()}, 数据类型: {image_tensor.dtype}")
            print("模型推理开始")
            logit = st.session_state.model(image_tensor)
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


def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


choice = st.sidebar.radio(label="What do you want to detect?", options=('Image'), index=0)

if choice == 'Image':
    # 使用session_state保存默认图片显示状态
    if 'show_default' not in st.session_state:
        st.session_state.show_default = False

    if st.button("📁 使用默认测试图片"):
        st.session_state.show_default = True

    if st.session_state.show_default:
        test_image_folder = './test/image'
        test_image_files = os.listdir(test_image_folder)
        if test_image_files:
            # 添加CSS样式
            st.markdown("""
            <style>
            .card {
                height: 300px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                margin-bottom: 20px;
                position: relative;
            }
            .card-img {
                flex: 1;
                display: flex;
                align-items: center;
                overflow: hidden;
                border-radius: 8px;
            }
            .card-img img {
                width: 100%;
                height: auto;
                object-fit: contain;
            }
            .card-footer {
                margin-top: auto;
                padding: 10px 0;
            }
            </style>
            """, unsafe_allow_html=True)

            # 创建3列布局
            cols = st.columns(3)

            for idx, img_file in enumerate(test_image_files):
                img_path = os.path.join(test_image_folder, img_file)
                with cols[idx % 3]:  # 自动循环使用3列
                    try:
                        img = Image.open(img_path).convert('RGB')

                        # 创建卡片式布局
                        st.markdown(
                            f'''
                            <div class="card">
                                <div class="card-img">
                                    <img src="data:image/png;base64,{image_to_base64(img)}">
                                </div>
                                <div class="card-footer">
                            ''',
                            unsafe_allow_html=True
                        )

                        # 在底部添加按钮
                        if st.button(f"选择 {img_file}", key=f"select_{idx}"):
                            st.session_state.selected_img = img_path

                        st.markdown('</div></div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"无法加载图片 {img_file}: {e}")

            if 'selected_img' in st.session_state and st.session_state.selected_img:
                st.success(f"已选择: {os.path.basename(st.session_state.selected_img)}")

                if st.button('​​**​​start to detect​​**​​', key="detect_default"):
                    try:
                        img = Image.open(st.session_state.selected_img).convert('RGB')
                        img_array = np.array(img)
                        image_tensor = preprocess(img_array)

                        if 'model_loaded' not in st.session_state:
                            model = models.resnet50(pretrained=False)
                            model.fc = torch.nn.Linear(2048, 2)
                            states = torch.load(f"{model_dir}/model1.pth",
                                                map_location=torch.device("cpu"))
                            states = states['model']
                            states = {key[2:]: value for key, value in states.items()}
                            model.load_state_dict(states)
                            model = model.to(device)
                            model.eval()
                            st.session_state.model = model
                            st.session_state.model_loaded = True

                        # 执行预测
                        prediction, confidence = predict_img(image_tensor)
                        st.info(f"📋the face in image is ​​**​​{prediction}​​**​​")
                        st.info(f"📋the confidence is ​​**​​{confidence:.2f}​​**​​")

                    except Exception as e:
                        st.error(f"检测出错: {e}")

    st.markdown("""
    <style>
        div[data-testid="stFileUploader"] label p {
            font-size: 24px !important;
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(label="​**​选择本地想要检测的的图片​**​", type=['jpg', 'png', 'jpeg'])

else:
    # 添加一个按钮用于选择默认测试视频
    if st.button("📁 使用默认测试视频"):
        test_video_folder = './test/video'
        test_video_files = os.listdir(test_video_folder)
        if test_video_files:
            # 默认选择第一个测试视频
            video_path = os.path.join(test_video_folder, test_video_files[0])
            uploaded_file = open(video_path, 'rb')
            # 预览默认视频的第一帧
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, caption='默认测试视频首帧')
            cap.release()
            uploaded_file.seek(0)  # 重置文件指针以便后续处理
    # 保留原始文件上传器
    uploaded_file = st.file_uploader(label="​**​选择要判断的视频​**​", type=['mp4', 'avi'])

# 显示结果
if uploaded_file is not None:
    if choice == 'Image':
        # 读取上传的图片
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='uploaded image')
        if 'model_loaded' not in st.session_state:
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(2048, 2)
            states = torch.load(f"{model_dir}/model1.pth", map_location=torch.device("cpu"))
            states = states['model']
            states = {key[2:]: value for key, value in states.items()}
            model.load_state_dict(states)
            model = model.to(device)
            model.eval()
            st.session_state.model = model
            st.session_state.model_loaded = True

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
                st.info(f"📋the face in image is **{prediction}**")
                st.info(f"📋the confidence is **{confidence}**")






