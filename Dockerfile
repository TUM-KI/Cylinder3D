FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-dev python3-pip
RUN apt install -y git
RUN apt install -y build-essential
RUN pip3 install torch==1.9.0
RUN pip3 install pyyaml==5.4.1
RUN pip3 install cython==0.29.24
RUN pip3 install nuscenes-devkit==1.1.6
RUN pip3 install numba==0.53.1
RUN pip3 install strictyaml==1.4.4

# Torch scatter
RUN pip3 install torch-scatter==2.0.8

# SpConv
RUN git clone --recursive https://github.com/traveller59/spconv.git /spconv
WORKDIR /spconv
RUN git checkout fad3000249d27ca918f2655ff73c41f39b0f3127
RUN apt install -y libboost-all-dev cmake
RUN pip3 install wheel==0.37.0 
RUN python3 setup.py bdist_wheel
WORKDIR /spconv/dist
RUN pip3 install spconv-1.2.1-cp38-cp38-linux_x86_64.whl

## Install Cylinder 3D
RUN git clone --recursive --depth 1 https://github.com/TUM-KI/Cylinder3D.git /cylinder3d
WORKDIR /cylinder3d


