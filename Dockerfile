# 使用官方 Python 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目的所有文件到容器中
COPY . /app

# 安装所需的系统依赖
RUN apt-get update && \
    apt-get install -y cmake build-essential git

# 安装 Python 依赖
RUN pip install -r requirements.txt

# 复制并执行自定义安装脚本
COPY install_dlib.sh /app/install_dlib.sh
RUN chmod +x /app/install_dlib.sh && /app/install_dlib.sh

# 启动 Streamlit 应用
CMD ["streamlit", "run", "Home.py"]
