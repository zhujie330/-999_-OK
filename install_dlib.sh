#!/bin/bash
apt-get update
apt-get install -y cmake build-essential
pip install cmake  # 以防万一
git clone https://github.com/davisking/dlib.git
cd dlib
python setup.py install
