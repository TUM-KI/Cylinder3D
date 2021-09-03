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

def transform_to_global(pc, lidar_transform):
    # ego pose * calibrated sensor
    # print(f"num points before global {pc.shape[1]} with shape {pc.shape}")
    # transform = lidar_transform[4:,:] @ lidar_transform[:4,:]
    # homogenous_points = to_homogenous_points(pc)
    # result = transform @ homogenous_points
    # transformed_points = pc.copy()
    # transformed_points[:3,:] = result[:3,:]
    # print(f"num points in global {transformed_points.shape[1]}")
    # return transformed_points

    new_pc = pc.copy()
    new_pc = new_pc.astype(np.float64)

    ego_rot, ego_trans, csr_rot, csr_trans = lidar_transform

    rotation = Quaternion(csr_rot).rotation_matrix #lidar_transform[:4,:][:3,:3]
    translation = csr_trans #lidar_transform[:4,:][:3,3]
    new_pc[:3,:] = np.dot(rotation, new_pc[:3,:])
    for i in range(3):
        new_pc[i,:] += translation[i]

    rotation = Quaternion(ego_rot).rotation_matrix #lidar_transform[4:,:][:3,:3]
    translation = ego_trans #lidar_transform[4:,:][:3,3]
    new_pc[:3,:] = np.dot(rotation, new_pc[:3,:])
    for i in range(3):
        new_pc[i,:] += translation[i]

    return new_pc

def transform_to_camera(pc, camera_transforms, colormap):
    # to image frame
    # print(f"num points before transform {pc.shape[1]} with shape {pc.shape}")
    # transformed_points = pc.copy()
    # homogenous_points = to_homogenous_points(pc)
    # transform = np.linalg.inv(camera_transforms[4:8,:]@camera_transforms[:4,:])
    # homogenous_points = transform @ homogenous_points
    # print(f"num points in frame {homogenous_points.shape[1]}")
    # # to camera
    # viewpad = np.eye(4)
    # viewpad[:3,:3] = camera_transforms[8:,:][:3,:3]
    # transformed_points[:3,:] = (viewpad @ homogenous_points)[:3,:]
    # transformed_points = transformed_points.astype(np.float64)
    # print(f"num points in camera {transformed_points.shape[1]}")
    # # normalize
    # transformed_points = transformed_points[:3,:]
    # nbr_points = transformed_points.shape[1]
    # print(nbr_points)
    # transformed_points = transformed_points / transformed_points[2:3,:].repeat(3,0).reshape(3, nbr_points)
    # return transformed_points
    new_pc = pc.copy()
    new_pc = new_pc.astype(np.float64)
    print(f"new pc shape {new_pc.shape}")

    ego_rot, ego_trans, csr_rot, csr_trans, intrins, to_lidar_rot, to_lidar_trans = camera_transforms

    translation = -ego_trans #-camera_transforms[4:8,:][:3,3]
    print(f"translation {translation}")
    rotation = Quaternion(ego_rot).rotation_matrix.T#camera_transforms[4:8,:][:3,:3].T
    for i in range(3):
        new_pc[i,:] += translation[i]
    new_pc[:3,:] = np.dot(rotation, new_pc[:3,:])

    translation = -csr_trans#-camera_transforms[:4,:][:3,3]
    rotation = Quaternion(csr_rot).rotation_matrix.T #camera_transforms[:4,:][:3,:3].T
    for i in range(3):
        new_pc[i,:] += translation[i]
    new_pc[:3,:] = np.dot(rotation, new_pc[:3,:])

    save_point_cloud('tmp/cam.ply', new_pc.T, colormap)

    # translation = to_lidar_trans#-camera_transforms[:4,:][:3,3]
    # rotation = to_lidar_rot #Quaternion(to_lidar_rot).rotation_matrix.T #camera_transforms[:4,:][:3,:3].T
    # new_pc[:3,:] = np.dot(rotation, new_pc[:3,:])
    # for i in range(3):
    #     new_pc[i,:] += translation[i]

    # depths = new_pc[2,:]
    # nbr_points = new_pc.shape[1]

    # new_pc = np.concatenate((new_pc, np.ones((1, nbr_points))))
    # viewpad = np.eye(4)
    # viewpad[:intrins.shape[0],:intrins.shape[1]] = intrins #camera_transforms[8:,:][:3,:3]
    # #print(camera_transforms[8:,:])
    # print(viewpad)
    # transformed_points = np.dot(viewpad, new_pc).astype(np.float64)
    # print(f"num points in camera {transformed_points.shape[1]}")
    # # normalize
    # transformed_points = transformed_points[:3,:]
    # nbr_points = transformed_points.shape[1]
    # print(nbr_points)
    # transformed_points = transformed_points / transformed_points[2:3,:].repeat(3,0).reshape(3, nbr_points)
    # return transformed_points
    return view_points(new_pc[:3,:], intrins, normalize=True)

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
    mask = np.logical_and(mask, lidar[0,:] > 1)
    mask = np.logical_and(mask, lidar[0,:] < image.shape[1] - 1)
    mask = np.logical_and(mask, lidar[1,:] > 1)
    mask = np.logical_and(mask, lidar[1,:] < image.shape[0] -1)
    return lidar[:, mask]

def render_lidar_into_image_stack(img_stack, pc, camera_transforms, colormap):
    camera_channel = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    for i in range(len(camera_channel)):
        image = img_stack[i]
        transforms = camera_transforms[i]
        tmp = pc.copy()
        lidar_in_camera = transform_to_camera(tmp, transforms, colormap)
        #Slidar_in_camera = filter_points_not_in_cam(lidar_in_camera, image)
        fig, ax = plt.subplots(1,1)
        ax.imshow(image)
        ax.scatter(lidar_in_camera[0,:], lidar_in_camera[1,:])
        ax.axis('off')
        plt.show()
        s, (width, height) = fig.canvas.print_to_buffer()
        img = np.fromstring(s, np.uint8).reshape((height, width, 4))
        img = Image.fromarray(img)
        img.save(f"tmp/frame_{camera_channel[i]}.png")

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
            print(f" voxel position shape: {voxel_position.shape}")
            
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

            for count, _ in enumerate(val_grid):
                label = predicted_labels[
                    count, val_grid[count][:,0], 
                    val_grid[count][:,1],
                    val_grid[count][:,2]
                ]
                print(f"tmp shape  {val_grid[count].shape}")
                pointcloud = polar2cat( val_grid[count].T ).T
                voxel_position_global = transform_to_global(voxel_position[0].numpy(), lidar_transforms)
                print(f"voxel position global {voxel_position_global}")
                # pointcloud = transform_to_global(pointcloud.T, lidar_transforms)
                groundtruth = val_pt_labs[count]

                predicted_colors = [label_colormap[i] for i in label]
                groundtruth_colors = [label_colormap[i] for i in groundtruth[:,0]]
                pointcloud_vis(f"tmp/{count}", pointcloud, label, groundtruth, label_colormap)
                pointcloud_vis(f"tmp/xyz", voxel_position_global, label, groundtruth, label_colormap)
                pointcloud_vis(f"tmp/xyz_untransformed", voxel_position[0].numpy(), label, groundtruth, label_colormap)
                render_lidar_into_image_stack(camera_images, voxel_position_global.T, camera_transforms, predicted_colors)
                break

            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/nuScenes.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)