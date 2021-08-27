from utils.load_save_util import load_checkpoint_1b1
import torch
import argparse
import os
import numpy as np
import sys

from config.config import load_config_data
from builder import data_builder, model_builder
from dataloader.pc_dataset import get_nuScenes_label_name, get_nuScenes_colormap


from plyfile import PlyData, PlyElement

def save_point_cloud(path, pc, color):
    num_vertices = pc.shape[0]
    vertices = np.zeros((num_vertices), dtype=[('x', np.float32), ('y', np.float32),
    ('z', np.float32), ('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte)])

    for index in range(pc.shape[0]):
        point = pc[index,:]
        vertices[index] = (*point, *color[index])
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], text=True).write(path)


def pointcloud_vis(pc, predicted_label, groundtruth_label, colormap):

    unique_label_prediction = np.unique(predicted_label)
    unique_label_groundtruth = np.unique(groundtruth_label[:,0])
    prediction_bin_count = np.bincount(predicted_label)[unique_label_prediction]
    groundtruth_bin_count = np.bincount(groundtruth_label[:,0])[unique_label_groundtruth]
    
    print(f"unique predicted {set(zip(unique_label_prediction, prediction_bin_count))}")
    print(f"unique groundtruth {set(zip(unique_label_groundtruth, groundtruth_bin_count))}")

    predicted_colors = [colormap[i] for i in predicted_label]
    groundtruth_colors = [colormap[i] for i in groundtruth_label[:,0]]

    save_point_cloud('tmp/predicted.ply', pc, predicted_colors)
    save_point_cloud('tmp/groundtruth.ply', pc, groundtruth_colors)


def main(args):
    pytorch_device = torch.device('cuda:0')

    configs = load_config_data(args.config_path)

    label_mapping = configs['dataset_params']['label_mapping']
    label_name = get_nuScenes_label_name(label_mapping)
    label_colormap = get_nuScenes_colormap(label_mapping)
    print(label_name)

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
            predicted_labels = torch.argmax(predicted_labels, dim=1)
            predicted_labels = predicted_labels.cpu().detach().numpy()
            # print(predicted_labels[0])

            #ego_transform, csr_transform = val_dataset_loader.get_transform
            print(f"shape of val_grid: {val_grid[0].shape}")

            for count, _ in enumerate(val_grid):
                label = predicted_labels[
                    count, val_grid[count][:,0], 
                    val_grid[count][:,1],
                    val_grid[count][:,2]
                ]
                pointcloud = val_grid[count]
                groundtruth = val_pt_labs[count]
                pointcloud_vis(pointcloud, label, groundtruth, label_colormap)
                break

            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuScenes.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)