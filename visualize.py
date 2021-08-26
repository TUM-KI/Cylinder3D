from utils.load_save_util import load_checkpoint_1b1
import torch
import argparse
import os
import numpy as np
import sys

from config.config import load_config_data
from builder import data_builder, model_builder

def main(args):
    pytorch_device = torch.device('cuda:0')

    configs = load_config_data(args.config_path)

    model_config = configs['model_params']
    my_model = model_builder.build(model_config)
    model_load_path = configs['train_params']['model_load_path']
    if os.path.exists(model_load_path):
        print("has model")
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)

    train_dataset_loader, val_dataset_loader = data_builder.build(
        configs['dataset_params'],
        configs['train_data_loader'],
        configs['val_data_loader'],
        grid_size=configs['model_params']['output_shape']
    )

    my_model.eval()
    val_batch_size = configs['val_data_loader']['batch_size']
    with torch.no_grad():
        for i, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(val_dataset_loader):
            val_pt_fea_tensor = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
            val_grid_tensor = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predicted_labels = my_model(val_pt_fea_tensor, val_grid_tensor, val_batch_size)
            print(predicted_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuScenes.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)