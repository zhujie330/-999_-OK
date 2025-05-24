import os
import tempfile
import streamlit as st
from modelscope.hub.snapshot_download import snapshot_download
st.write("⚠️ 由于 Git LFS 流量已达上线，自动转从 ModelScope 联网加载模型，")
@st.cache_resource(show_spinner="加载模型中，请稍候...")
def get_model_dir():
    model_dir = os.path.join(tempfile.gettempdir(), 'model_use414')
    model_file_path = os.path.join(model_dir, 'model1.pth')

    if not os.path.exists(model_file_path):
        model_dir = snapshot_download('zhujie67o/model_use414')
    return model_dir
