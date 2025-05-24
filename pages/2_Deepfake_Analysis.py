import os

import face_recognition
import numpy as np
import streamlit as st
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import models, transforms
from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.v2.functional import to_pil_image
from modelscope import snapshot_download
from draw_gradient import compute_gradient, visualize_heatmap
from saliency.gradcam import GradCAM
import tempfile
import os
from utils_model import get_model_dir 



st.set_page_config(page_title="Deepfake Detection", page_icon="🔬")
st.sidebar.header("🔬Deepfake Detection")
st.write("# Demo for Deepfake Analysis🔬")
st.write("⚠️ 由于 Git LFS 流量已达上线，自动转从 ModelScope 联网加载模型，请稍后")

model_dir = get_model_dir()
model_file_path = os.path.join(model_dir, 'model1.pth')
if os.path.exists(model_file_path):
    st.write("✔️ 模型已加载")
else:
    st.write("⚠️ 模型文件未找到，请稍候重试")


    st.write("✔️ 模型已加载")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载 OpenMP





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


def show_feature_map(layer_index):
    if layer_index == 4:
        model_new = torch.nn.Sequential(*(list(model.children())[:-2]))
    else:
        layer = None
        for name, module in model.named_children():
            if name == f"layer{layer_index+1}":
                layer = module
                break
        if layer is None:
            raise ValueError(f"Layer not found in the model.")
        model_new = torch.nn.Sequential(*(list(model.children())[:list(model.children()).index(layer)]))
    # model_new = torch.nn.Sequential(*(list(model.children())[:-2]))
    model_new = model_new.to(device)
    image_tensor = preprocess(img_array)
    feature_map = model_new(image_tensor.to(device))
    # 创建一个新的 Matplotlib 图形
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    st.text(f'feature map after layer{layer_index}')
    # 循环遍历每个特征图并将其添加到 Matplotlib 图形中
    for i in range(64):
        row = i // 8
        col = i % 8
        axs[row, col].imshow(feature_map[0, i].cpu().detach().numpy(), cmap='viridis')
        axs[row, col].axis('off')
    # 显示 Matplotlib 图形在 Streamlit 中
    st.pyplot(fig)


# .streamlit
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
# model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(2048, 2)
# states = torch.load(
#     os.path.join("D:\\其他\\wehchatfile\\WeChat Files\\wxid_3hhhdkir3jfj22\\FileStorage\\File\\2024-07", "model1.pth"))

#states = torch.load("./model1.pth", map_location=torch.device("cpu"))
states = torch.load(f"{model_dir}/model1.pth", map_location=torch.device("cpu"))
states = states['model']
states = {key[2:]: value for key, value in states.items()}
model.load_state_dict(states)
model = model.to(device)
model.eval()



map = st.sidebar.radio(
    label="Which would you like to be observe?",
    options=("Feature Map", "Saliency Map", "Class Activation Map"), index=None
)
uploaded_file = st.file_uploader(label="**choose the image you want to analyze**", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='uploaded image', use_column_width=True)

    img_array = np.array(img)
    image_tensor = preprocess(img_array)

    if map == "Feature Map":
        col1, col2, col3 = st.columns(3)

        with col1:
            show_feature_map(2)

        with col2:
            show_feature_map(3)

        with col3:
            show_feature_map(4)

    elif map == "Saliency Map":
        # 计算梯度
        gradient = compute_gradient(image_tensor.to(device), model)

        # 可视化热度图
        heatmap, saliency = visualize_heatmap(gradient)

        # 将输入图像转换成 numpy 数组
        input_image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        # 将热度图和输入图像相加
        superimposed_img = heatmap + input_image_np

        # 归一化相加后的图像
        superimposed_img /= np.max(superimposed_img)

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots()
            ax.imshow(input_image_np)
            ax.set_title('image')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.imshow(saliency)
            ax.set_title("saliency map")
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots()
            ax.imshow(superimposed_img)
            ax.set_title("saliency map with image")
            st.pyplot(fig)

    elif map == "Class Activation Map":
        with SmoothGradCAMpp(model) as cam_extractor:
            # Preprocess your data and feed it to the model
            out = model(image_tensor.to(device))
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        # Resize the CAM and overlay it
        result = overlay_mask(to_pil_image((image_tensor.squeeze(0)*255).to(torch.uint8)),
                              to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        # 循环遍历每个特征图并将其添加到 Matplotlib 图形中
        input_image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots()
            ax.imshow(input_image_np)
            ax.set_title("input image")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.imshow(activation_map[0].squeeze(0).detach().cpu().numpy())
            ax.set_title("class activation map")
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots()
            ax.imshow(result)
            ax.set_title("class activation map on image")
            st.pyplot(fig)



