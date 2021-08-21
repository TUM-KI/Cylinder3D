import numpy as np
import os
import torch
import argparse
from tqdm import tqdm

from dataloader.pc_dataset import get_nuScenes_label_name
from utils.load_save_util import load_checkpoint, load_checkpoint_1b1
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.metric_util import per_class_iu, fast_hist_crop

from typing import Dict

def main(args: Dict) -> None:
    pytorch_device = torch.device('cuda:0')

    configs = load_config_data(args.config_path)

    dataset_config = configs['dataset_params']
    model_config = configs['model_params']
    train_hypers = configs['train_params']

    val_dataloader_config = configs['val_data_loader']

    label_name = get_nuScenes_label_name(dataset_config['label_mapping'])
    unique_label = np.asarray(sorted(list(label_name.keys())))[1:] - 1
    unique_label_str = [label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(train_hypers['model_load_path']):
        print("has model checkpoint")
        my_model = load_checkpoint_1b1(
            train_hypers['model_load_path'],
            my_model
        )
    else:
        print("no model checkpoint")
    my_model.to(pytorch_device)

    _, val_dataset_loader = data_builder.build(
        dataset_config, configs['train_data_loader'],
        configs['val_data_loader'], grid_size=model_config['output_shape']
    )

    pbar = tqdm(total=len(val_dataset_loader))
    my_model.eval()
    hist_list = []
    with torch.no_grad():
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(val_dataset_loader):
            val_pt_fea_ten = [
                torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device)
                for i in val_pt_fea
            ]
            val_grid_ten = [
                torch.from_numpy(i).to(pytorch_device) 
                for i in val_grid
            ]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_dataloader_config['batch_size'])
            predict_labels = torch.argmax(predict_labels, dim=1).cpu().detach().numpy()
            
            for count, i_test_grid in enumerate(val_grid):
                hist_list.append(
                    fast_hist_crop(predict_labels[
                        count, val_grid[count][:,0],
                        val_grid[count][:,1],
                        val_grid[count][:,2]],
                        val_pt_labs[count], unique_label
                    )
                )

            pbar.update(1)

    iou = per_class_iu(sum(hist_list))
    print("Validation per class iou: ")
    for class_name, class_iou in zip(unique_label_str, iou):
        print(f"{class_name}: {class_iou*100}")
    val_miou = np.nanmean(iou) * 100
    print(f"Val Miou: {val_miou}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--config-path', default='config/nuScenes.yaml')
    args = parser.parse_args()
    main(args)
