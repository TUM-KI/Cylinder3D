#!/bin/bash
DATA_FOLDER=/data/dataset/nuScenes
SAVE_FOlDER=/data/dataset/results

python3 demo_folder.py --demo-folder $DATA_FOLDER --save-folder $SAVE_FOlDER \
    --config_path config/nuScenes.yaml 