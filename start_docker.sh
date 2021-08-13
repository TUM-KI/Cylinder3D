#!/bin/bash
distroname="$(lsb_release -s -d)"
datafolder=/home/derkacz/ki-al/datasets/nuscenes/new
resultfolder=/home/derkacz/ki-al/datasets/results
modelfolder=/home/derkacz/ki-al/datasets/nuscenes
containername=cylinder3d
manjaro="Manjaro Linux"

docker build -t $containername .
# For Manjaro use this:
docker run --rm --gpus all -it --privileged -v /dev:/dev -v $datafolder:/data/dataset/nuScenes --shm-size 16G \
    -v $resultfolder:/data/datasets/results -v $modelfolder:/cylinder3d/model_load_dir_nuscenes $containername python3 test.py

# For Ubuntu 20 use this
#docker run --rm --gpus all -it -v $datafolder:/data/dataset/nuScenes --shm-size 16G $containername bash 

# For Ubuntu 16 use this
# docker run --rm --runtime=nvidia -it -v $datafolder:/data/dataset/nuScenes --shm-size 16G $containername bash
