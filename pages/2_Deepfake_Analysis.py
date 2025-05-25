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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def preprocess(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    faces = face_recognition.face_locations(img)
    top, right, bottom, left = faces[0]
    face_img = img[top:bottom, left:right, :]
    face_img = torch.tensor(face_img / 255.0).permute(2, 0, 1)
    image_tensor = transform(face_img).unsqueeze(0)
    return image_tensor

def show_feature_map(layer_index, image_tensor):
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
    model_new = model_new.to(device)
    feature_map = model_new(image_tensor.to(device))
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    st.text(f'feature map after layer{layer_index}')
    for i in range(64):
        row = i // 8
        col = i % 8
        axs[row, col].imshow(feature_map[0, i].cpu().detach().numpy(), cmap='viridis')
        axs[row, col].axis('off')
    st.pyplot(fig)

device = torch.device('cpu')
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(2048, 2)
states = torch.load(f"{model_dir}/model1.pth", map_location=device)
states = states['model']
states = {key[2:]: value for key, value in states.items()}
model.load_state_dict(states)
model = model.to(device)
model.eval()

map = st.sidebar.radio(
    label="Which would you like to observe?",
    options=("Feature Map", "Saliency Map", "Class Activation Map"), index=None
)

# 显示默认测试图片和上传入口
if 'show_default' not in st.session_state:
    st.session_state.show_default = False
if 'selected_img' not in st.session_state:
    st.session_state.selected_img = None

st.markdown("## 使用系统测试图片")
if st.button("📁 显示系统测试图片"):
    st.session_state.show_default = True

img_array = None
image_tensor = None

if st.session_state.show_default:
    test_image_folder = './test/image'
    test_image_files = os.listdir(test_image_folder)

    if test_image_files:
        cols = st.columns(3)
        for idx, img_file in enumerate(test_image_files):
            img_path = os.path.join(test_image_folder, img_file)
            with cols[idx % 3]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    st.image(img, caption=img_file, use_container_width=True)
                    if st.button(f"选择 {img_file}", key=f"select_{idx}"):
                        st.session_state.selected_img = img_path
                        st.success(f"已选择：{img_file}")
                except Exception as e:
                    st.error(f"无法加载图片: {e}")

if st.session_state.selected_img:
    img = Image.open(st.session_state.selected_img).convert("RGB")
    st.image(img, caption='选中的测试图片', use_column_width=True)
    img_array = np.array(img)
    image_tensor = preprocess(img_array)

st.markdown("## 上传你自己的图片进行分析")
uploaded_file = st.file_uploader(label="**选择你要分析的图片**", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='上传的图片', use_column_width=True)
    img_array = np.array(img)
    image_tensor = preprocess(img_array)

if image_tensor is not None:
    if map == "Feature Map":
        col1, col2, col3 = st.columns(3)
        with col1:
            show_feature_map(2, image_tensor)
        with col2:
            show_feature_map(3, image_tensor)
        with col3:
            show_feature_map(4, image_tensor)

    elif map == "Saliency Map":
        gradient = compute_gradient(image_tensor.to(device), model)
        heatmap, saliency = visualize_heatmap(gradient)
        input_image_np = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        superimposed_img = heatmap + input_image_np
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
            out = model(image_tensor.to(device))
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        result = overlay_mask(to_pil_image((image_tensor.squeeze(0)*255).to(torch.uint8)),
                              to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
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
