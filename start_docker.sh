#!/bin/bash
distroname="$(lsb_release -s -d)"
datafolder=/home/daniel/projects/kidt/dataset/original
containername=cylinder3d
manjaro="Manjaro Linux"

docker build -t $containername .
echo $distroname
#if [[ $distroname == $manjaro ]]; then 
    docker run --rm --gpus all -it --privileged -v /dev:/dev -v $datafolder $containername bash
#else
#    docker run --rm --gpus all -it -v $datafolder $containername bash 
#fi
