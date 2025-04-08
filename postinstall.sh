#!/bin/bash
# 强制使用Bash严格模式
set -eo pipefail

# 安装系统级依赖（新增图形库和开发工具链）
apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev

# 设置dlib编译参数（启用CPU加速指令集）
export USE_AVX_INSTRUCTIONS=1
export DLIB_USE_CUDA=0  # 强制禁用CUDA避免环境冲突

# 通过源码编译安装dlib
pip install --no-cache-dir --force-reinstall \
    --global-option="build_ext" \
    --global-option="--no USE_AVX_INSTRUCTIONS" \
    dlib==19.22.0
