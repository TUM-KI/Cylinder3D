FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-dev python3-pip
RUN pip3 install --upgrade pip
RUN apt install -y git wget
RUN apt install -y build-essential

# get newer version of cmake
RUN wget https://apt.kitware.com/kitware-archive.sh
RUN chmod +x kitware-archive.sh && ./kitware-archive.sh

#RUN pip3 install torch
RUN pip3 install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pyyaml==5.4.1
RUN pip3 install cython==0.29.24
RUN pip3 install nuscenes-devkit==1.1.6
RUN pip3 install numba==0.53.1
RUN pip3 install strictyaml==1.4.4

# Torch scatter
#RUN pip3 install torch-scatter
RUN pip3 install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html

# SpConv
RUN git clone --recursive https://github.com/traveller59/spconv.git /spconv
WORKDIR /spconv
RUN git checkout fad3000249d27ca918f2655ff73c41f39b0f3127
RUN apt install -y libboost-all-dev cmake
RUN pip3 install wheel
RUN python3 setup.py bdist_wheel
WORKDIR /spconv/dist
RUN pip3 install *.whl

## Install Cylinder 3D
RUN git clone --recursive --depth 1 https://github.com/TUM-KI/Cylinder3D.git /cylinder3d
WORKDIR /cylinder3d
RUN chmod +x train_nusc.sh


