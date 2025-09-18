import os  # noqa
import json
import pickle
import warnings
from typing import Union, List, Tuple
import argparse
import numpy as np
import pdal
import torch
from omegaconf import DictConfig
from sklearn.neighbors import KDTree
from torch import LongTensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def execuate_las(raw_path, grid_size, split, output_path, tile_size = 250.0, has_label=True, tile_orient=True):
    scene_name = os.path.splitext(os.path.basename(raw_path))[0]

    has_intensity = has_color = True
    pipeline = pdal.Pipeline()
    pipeline |= pdal.Reader.las(filename=raw_path)
    pipeline |= pdal.Filter.stats(dimensions="Intensity,Red,Blue,Green")
    pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=grid_size)
    if has_label:
        pipeline |= pdal.Filter.range(limits="Classification[2:6], Classification[8:8]")
        pipeline |= pdal.Filter.assign(value=[
            f"Classification = 0 WHERE Classification == 2",
            f"Classification = 1 WHERE ((Classification >= 3) && (Classification <= 5))",
            f"Classification = 2 WHERE Classification == 6",
            f"Classification = 3 WHERE Classification == 8"
        ])
    else:
        print("No label, only process point cloud.")
    count = pipeline.execute()
    arrays = pipeline.arrays[0]
    metadata = pipeline.metadata['metadata']
    bounds = metadata['readers.las']
    minx = bounds['minx']
    miny = bounds['miny']
    minz = bounds['minz']
    pos = np.concatenate([
        np.expand_dims(arrays['X'] - minx, 1),
        np.expand_dims(arrays['Y'] - miny, 1),
        np.expand_dims(arrays['Z'] - minz, 1)
    ], axis=-1).astype(np.float32)
    y = arrays['Classification'].reshape(-1, 1)

    stats = metadata['filters.stats']['statistic'][1]
    r = normalize_attributes(arrays['Red'], 0.0, (float)(stats['maximum']), stats['average'],
                                  stats['stddev'])
    stats = metadata['filters.stats']['statistic'][2]
    g = normalize_attributes(arrays['Green'], 0.0, (float)(stats['maximum']), stats['average'],
                                  stats['stddev'])
    stats = metadata['filters.stats']['statistic'][3]
    b = normalize_attributes(arrays['Blue'], 0.0, (float)(stats['maximum']), stats['average'],
                                  stats['stddev'])
    color = np.concatenate([r, g, b], axis=-1)


    #如果 tile_size 大于 0，则将点云分割成小块,每个小块对应单独的save_path (例如：train_data_01, train_data_02)
    if tile_size > 0:
        if tile_orient:
            print("start tiling with oriented bounding box...")
            save_path = os.path.join(output_path, split)
            os.makedirs(save_path, exist_ok=True)

            # 1. 点云采样策略 - 根据点数动态调整采样率
            sample_size = 50000  # 目标采样点数
            if len(pos) > sample_size:
                sample_ratio = sample_size / len(pos)
                indices = np.random.choice(len(pos), size=sample_size, replace=False)
                sampled_pos = pos[indices]
            else:
                sampled_pos = pos

            # 2. 计算主方向（PCA）
            centroid = np.mean(sampled_pos, axis=0)
            centered = sampled_pos - centroid
            cov = centered.T @ centered / (len(sampled_pos) - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # 确保正交基的右手法则（防止旋转镜像）
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, -1] = -eigenvectors[:, -1]

            # 根据特征值降序排列特征向量
            sort_indices = np.argsort(eigenvalues)[::-1]
            R = eigenvectors[:, sort_indices].T

            # 3. 将整个点云旋转到主方向坐标系
            rotated_pos = (R @ (pos - centroid).T).T

            # 4. 计算旋转后点云的边界（忽略z方向）
            min_vals = rotated_pos.min(axis=0)
            max_vals = rotated_pos.max(axis=0)

            # 5. 在旋转坐标系中切片
            num_tiles_x = int(np.ceil((max_vals[0] - min_vals[0]) / tile_size))
            num_tiles_y = int(np.ceil((max_vals[1] - min_vals[1]) / tile_size))

            # 保存全局旋转信息（后续还原用）


            # 6. 高效分块处理（避免内存复制）
            for i in range(num_tiles_x):
                x_start = min_vals[0] + i * tile_size
                x_end = min_vals[0] + (i + 1) * tile_size

                for j in range(num_tiles_y):
                    y_start = min_vals[1] + j * tile_size
                    y_end = min_vals[1] + (j + 1) * tile_size

                    # 使用布尔索引创建掩码
                    tile_mask = (
                            (rotated_pos[:, 0] >= x_start) &
                            (rotated_pos[:, 0] < x_end) &
                            (rotated_pos[:, 1] >= y_start) &
                            (rotated_pos[:, 1] < y_end)
                    )

                    tile_count = np.count_nonzero(tile_mask)
                    if tile_count > 0:
                        tile_save_path = os.path.join(save_path, f"{scene_name}_tile_{i}_{j}")
                        os.makedirs(tile_save_path, exist_ok=True)

                        # 直接使用掩码保存切片
                        np.save(os.path.join(tile_save_path, "coord.npy"),
                                rotated_pos[tile_mask].astype(np.float32))
                        np.save(os.path.join(tile_save_path, "color.npy"),
                                color[tile_mask].astype(np.float32))
                        np.save(os.path.join(tile_save_path, "segment.npy"),
                                y[tile_mask].astype(np.int16))

        else:
            print("start tiling ...")
            save_path = os.path.join(output_path, split)
            os.makedirs(save_path, exist_ok=True)
            num_tiles_x = int(np.ceil((pos[:, 0].max() - pos[:, 0].min()) / tile_size))
            num_tiles_y = int(np.ceil((pos[:, 1].max() - pos[:, 1].min()) / tile_size))
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    tile_mask = (
                        (pos[:, 0] >= i * tile_size) & (pos[:, 0] < (i + 1) * tile_size) &
                        (pos[:, 1] >= j * tile_size) & (pos[:, 1] < (j + 1) * tile_size)
                    )
                    if np.any(tile_mask):
                        tile_pos = pos[tile_mask]
                        tile_color = color[tile_mask]
                        tile_y = y[tile_mask]
                        tile_save_path = os.path.join(save_path, f"{scene_name}_tile_{i}_{j}")
                        os.makedirs(tile_save_path, exist_ok=True)
                        np.save(os.path.join(tile_save_path, "coord.npy"), tile_pos.astype(np.float32))
                        np.save(os.path.join(tile_save_path, "color.npy"), tile_color.astype(np.float32))
                        np.save(os.path.join(tile_save_path, "segment.npy"), tile_y.astype(np.int16))
    else:
        save_path = os.path.join(output_path, split, scene_name)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "coord.npy"), pos.astype(np.float32))
        np.save(os.path.join(save_path, "color.npy"), color.astype(np.float32))
        np.save(os.path.join(save_path, "segment.npy"), y.astype(np.int16))
    # np.save(os.path.join(save_path, "instance.npy"), room_instance_gt.astype(np.int16))


def normalize_attributes(array: np.array, min_value: float, max_value: float, mean_value: float, std_value: float):
    """根据均值和标准差进行归一化操作，采用2倍标准差归一化

    Args:
        array (np.array): 输入数据
        min_value (float): 初始最小值
        max_value (float): 初始最大值
        mean_value (float): 均值
        std_value (float): 标准差

    Returns:
        object (np.array): 归一化，并且维度变成2维，方便 cat

    """
    min_value = np.clip(mean_value - 2 * std_value, min_value, max_value)
    max_value = np.clip(mean_value + 2 * std_value, min_value, max_value)
    value = (array.astype(np.float32) - min_value) / (max_value - min_value)
    value = np.clip(value, 0.0, 1.0)
    return np.expand_dims(value, 1)


def parse_lidar(dataset_root, grid_size, tile_size, has_label, tile_orient):
    print("Reading lidar files...")

    source_dir = os.path.join(dataset_root, 'raw')
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    save_path = os.path.join(dataset_root, 'processed')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    '''
        dataset_root：
        - raw
          -- train
          --- *.laz
          -- val
          --- *.laz
          -- test
          --- *.laz
        - processed
          -- train
          --- train_data_01
          ---- coord.npy
          ---- color.npy
          ---- segment.npy
          -- val
          --- val_data_01
          ---- coord.npy
          ---- color.npy
          ---- segment.npy
          -- test
          --- test_data_01
          ---- coord.npy
          ---- color.npy
          ---- segment.npy
        '''
    # 获取 'raw' 目录下的第二级文件夹名称
    split = []
    second_level_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for folder in second_level_dirs:
        split.append(folder)  # 将第二级文件夹名字保存到 'split'

        # 构建路径到最后一级（即 .laz 文件所在的文件夹）
        folder_path = os.path.join(source_dir, folder)

        # 循环处理 .laz 文件
        for laz_file in os.listdir(folder_path):
            if laz_file.endswith(".las" ) or laz_file.endswith(".laz"):  # 确保只处理 .las 和.laz 的文件
                print('start', laz_file)
                laz_file_path = os.path.join(folder_path, laz_file)
                # 调用处理函数，并传入 split
                execuate_las(laz_file_path, grid_size, folder, save_path, tile_size, has_label, tile_orient)


def main_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", help="Path where raw datasets are located.", default="/datasets/paper/"
    )
    # "/datasets/internship/unused_land_data"
    # "/datasets/ft_data/"
    parser.add_argument(
        "--grid_size", default=0.1, type=float, help="grid size in meters."
    )
    parser.add_argument(
        "--tile_size", default=-1, type=float, help="tile size in meters."
    )
    parser.add_argument(
        "--has_label", default=True, type=bool, help="has label."
    )
    parser.add_argument(
        "--tile_orient", default=True, type=bool, help="tile orientation."
    )
    args = parser.parse_args()

    print("Loading LAS information ...")

    parse_lidar(args.dataset_root, args.grid_size, args.tile_size, args.has_label, args.tile_orient)


if __name__ == "__main__":
    # main_process()
    main_preprocess()

