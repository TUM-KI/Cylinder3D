#!/bin/bash
distroname="$(lsb_release -s -d)"
datafolder=/home/derkacz/ki-al/datasets/nuscenes/new
containername=cylinder3d
manjaro="Manjaro Linux"

docker build -t $containername .
#echo $distroname
#if [[ $distroname == $manjaro ]]; then 
#    docker run --rm --gpus all -it --privileged -v /dev:/dev -v $datafolder:/data/dataset/nuScenes $containername bash
#else
    docker run --rm --gpus all -it -v $datafolder:/data/dataset/nuScenes --shm-size 16G $containername bash 
#fi
