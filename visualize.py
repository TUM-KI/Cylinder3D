from utils.load_save_util import load_checkpoint_1b1
import torch
import argparse
import os
import numpy as np
import sys

from config.config import load_config_data
from builder import data_builder, model_builder
from dataloader.pc_dataset import get_nuScenes_label_name, get_nuScenes_colormap

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image

from plyfile import PlyData, PlyElement
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

from dataloader.dataset_nuscenes import polar2cat

def save_point_cloud(path, pc, color):
    num_vertices = pc.shape[0]
    vertices = np.zeros((num_vertices), dtype=[('x', np.float32), ('y', np.float32),
    ('z', np.float32), ('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte)])

    for index in range(pc.shape[0]):
        point = pc[index,:]
        vertices[index] = (*point, *color[index])
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], text=True).write(path)


def pointcloud_vis(dest, pc, predicted_label, groundtruth_label, colormap):

    if not os.path.exists(dest):
        os.mkdir(dest)

    unique_label_prediction = sorted(np.unique(predicted_label))
    unique_label_groundtruth = sorted(np.unique(groundtruth_label[:,0]))
    prediction_bin_count = np.bincount(predicted_label)[unique_label_prediction]
    groundtruth_bin_count = np.bincount(groundtruth_label[:,0])[unique_label_groundtruth]
    
    print(f"unique predicted {set(zip(unique_label_prediction, prediction_bin_count))}")
    print(f"unique groundtruth {set(zip(unique_label_groundtruth, groundtruth_bin_count))}")

    predicted_colors = [colormap[i] for i in predicted_label]
    groundtruth_colors = [colormap[i] for i in groundtruth_label[:,0]]

    diff_labels = predicted_label == groundtruth_label[:,0]
    diff_colormap = {
        True: [135, 206, 235],  # skyblue
        False: [220, 20, 60]    # crimson
    }
    diff_colors = [diff_colormap[i] for i in diff_labels]

    save_point_cloud(f'{dest}/predicted.ply', pc, predicted_colors)
    save_point_cloud(f'{dest}/groundtruth.ply', pc, groundtruth_colors)
    save_point_cloud(f'{dest}/diff.ply', pc, diff_colors)

def to_homogenous_points(pc):
    shape = pc.shape
    return np.vstack((pc[:3,:], np.ones(shape[1]))) 

def to_4x4_matrix(rotation, translation):
    matrix = np.identity(4)
    matrix[:3,:3] = rotation
    matrix[:3,3] = translation
    return matrix

def transform_to_global(pc, transforms):
    ego_rot, ego_trans, csr_rot, csr_trans = transforms
    csr_transform = to_4x4_matrix(Quaternion(csr_rot).rotation_matrix, csr_trans)
    ego_transform = to_4x4_matrix(Quaternion(ego_rot).rotation_matrix, ego_trans)
    return (ego_transform @ csr_transform @ to_homogenous_points(pc))[:3, :]

def transform_to_camera(pc, transforms):
    ego_rot, ego_trans, csr_rot, csr_trans, intrinsics, _, _ = transforms
    csr_transform = to_4x4_matrix(Quaternion(csr_rot).rotation_matrix, csr_trans)
    ego_transform = to_4x4_matrix(Quaternion(ego_rot).rotation_matrix, ego_trans)
    inv_transform = np.linalg.inv(ego_transform @ csr_transform)
    pc_homogenous = to_homogenous_points(pc)
    pc_homogenous = inv_transform @ pc_homogenous

    viewpad = np.eye(4)
    viewpad[:3,:3] = intrinsics
    pc_homogenous = viewpad @ pc_homogenous
    new_pc = pc_homogenous[:3,:]
    nbr_points = new_pc.shape[1]
    new_pc /= new_pc[2:3,:].repeat(3,0).reshape(3,nbr_points)
    return new_pc

def filter_points_not_in_cam(lidar: np.ndarray, image: np.ndarray, min_dist: int = 1.0) -> np.ndarray:
    """
    Remove all points that are not directly in the image or 
    are to near to the camera lens.
    :param lidar: pointcloud
    :param image: the image
    :param min_dist: the minimum distance a point need to have from the camera
    :return: the filtered pointcloud
    """
    depths = lidar[2,:]
    print(f"image shape {image.shape}")
    print(f" in filter {lidar.shape}")
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    # mask = np.logical_and(mask, lidar[0,:] > 1)
    mask = np.logical_and(mask, lidar[0,:] < image.shape[1] - 1)
    # mask = np.logical_and(mask, lidar[1,:] > 1)
    mask = np.logical_and(mask, lidar[1,:] < image.shape[0] - 1)
    return lidar[:, mask]

def render_lidar_into_image_stack(img_stack, pc, camera_transforms, colormap):
    camera_channel = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    for i in range(len(camera_channel)):
        image = img_stack[i]
        transforms = camera_transforms[i]
        pointcloud = transform_to_camera(pc, transforms)
        #pointcloud = filter_points_not_in_cam(pointcloud, image)

        fig, ax = plt.subplots(1,1)
        ax.imshow(image)
        ax.scatter(pointcloud[0,:], pointcloud[1,:])
        ax.axis('off')
        plt.show()


def main(args):
    pytorch_device = torch.device('cuda:0')

    configs = load_config_data(args.config_path)

    label_mapping = configs['dataset_params']['label_mapping']
    label_name = get_nuScenes_label_name(label_mapping)
    label_colormap = get_nuScenes_colormap(label_mapping)

    model_config = configs['model_params']
    my_model = model_builder.build(model_config)
    model_load_path = configs['train_params']['model_load_path']
    if os.path.exists(model_load_path):
        print("has model")
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)

    train_dataset_loader, val_dataset_loader, train_dataset, val_dataset = data_builder.build(
        configs['dataset_params'],
        configs['train_data_loader'],
        configs['val_data_loader'],
        grid_size=configs['model_params']['output_shape'],
        return_dataset=True
    )

    my_model.eval()
    val_batch_size = configs['val_data_loader']['batch_size']
    with torch.no_grad():
        for i, (voxel_position, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(val_dataset_loader):
            
            
            val_pt_fea_tensor = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
            val_grid_tensor = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predicted_labels = my_model(val_pt_fea_tensor, val_grid_tensor, val_batch_size)
            predicted_labels = torch.argmax(predicted_labels, dim=1)
            predicted_labels = predicted_labels.cpu().detach().numpy()
            # print(predicted_labels[0])

            lidar_transforms, camera_transforms = val_dataset.get_transform()
            camera_images = val_dataset.get_images()

            #-------------------------------------
            # Test Workspace

            predicted_labels_colors = [label_colormap[i] for i in predicted_labels[
                                            0, val_grid[0][:,0], 
                                            val_grid[0][:,1],
                                            val_grid[0][:,2]
                                        ]]
            groundtruth_labels_colors = [label_colormap[i] for i in val_pt_labs[0][:,0]]
            pointcloud = voxel_position[0].numpy()
            # save the pointcloud in (hopefully) lidar sensor space
            save_point_cloud('tmp/lidar_sensor.predicted.ply', pointcloud, predicted_labels_colors)
            save_point_cloud('tmp/lidar_sensor.groundtruth.ply', pointcloud, groundtruth_labels_colors)

            pointcloud_global = transform_to_global(pointcloud.T, lidar_transforms).T
            # save the pointcloud in (hopefully) global space
            save_point_cloud('tmp/global.predicted.ply', pointcloud_global, predicted_labels_colors)
            save_point_cloud('tmp/global.groundtruth.ply', pointcloud_global, groundtruth_labels_colors)

            pointcloud_camera = transform_to_camera(pointcloud_global.T, camera_transforms[0]).T
            # save the pointcloud in (hopefully) frame space
            save_point_cloud('tmp/frame.predicted.ply', pointcloud_camera, predicted_labels_colors)
            save_point_cloud('tmp/frame.groundtruth.ply', pointcloud_camera, groundtruth_labels_colors)
            render_lidar_into_image_stack(camera_images, pointcloud_global.T, camera_transforms, label_colormap)

            # End: Test Workspace
            #-------------------------------------

            for count, _ in enumerate(val_grid):
                label = predicted_labels[
                    count, val_grid[count][:,0], 
                    val_grid[count][:,1],
                    val_grid[count][:,2]
                ]
                # print(f"tmp shape  {val_grid[count].shape}")
                # pointcloud = polar2cat( val_grid[count].T ).T
                # voxel_position_global = transform_to_global(voxel_position[0].numpy(), lidar_transforms)
                # print(f"voxel position global {voxel_position_global}")
                # # pointcloud = transform_to_global(pointcloud.T, lidar_transforms)
                # groundtruth = val_pt_labs[count]

                # predicted_colors = [label_colormap[i] for i in label]
                # groundtruth_colors = [label_colormap[i] for i in groundtruth[:,0]]
                # pointcloud_vis(f"tmp/{count}", pointcloud, label, groundtruth, label_colormap)
                # pointcloud_vis(f"tmp/xyz", voxel_position_global, label, groundtruth, label_colormap)
                # pointcloud_vis(f"tmp/xyz_untransformed", voxel_position[0].numpy(), label, groundtruth, label_colormap)
                # render_lidar_into_image_stack(camera_images, voxel_position_global.T, camera_transforms, predicted_colors)
                break

            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuScenes.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)