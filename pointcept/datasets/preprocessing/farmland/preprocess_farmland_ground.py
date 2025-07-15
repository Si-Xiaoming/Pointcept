import os # noqa
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

def execuate_las(raw_path, grid_size,split, output_path):
    scene_name = os.path.splitext(os.path.basename(raw_path))[0]


    has_intensity = has_color = True
    pipeline = pdal.Pipeline()
    pipeline |= pdal.Reader.las(filename=raw_path)
    pipeline |= pdal.Filter.stats(dimensions="Intensity")
    pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=grid_size)
    # pipeline |= pdal.Filter.range(limits="Classification[2:6], Classification[8:8]")

    # pipeline |= pdal.Filter.assign(value=[
    #     f"Classification = 0 WHERE Classification == 2",
    #     f"Classification = 1 WHERE ((Classification >= 3) && (Classification <= 5))",
    #     f"Classification = 2 WHERE Classification == 6",
    #     f"Classification = 3 WHERE Classification == 8"
    # ])

    pipeline |= pdal.Filter.assign(value=[
        f"Classification = 0 WHERE Classification != 2",
        f"Classification = 1 WHERE Classification == 2"
    ])

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


    stats = metadata['filters.stats']['statistic'][0]
    intensity = normalize_attributes(arrays['Intensity'], 0.0, (float)(stats['maximum']), stats['average'],
                                  stats['stddev'])




    save_path = os.path.join(output_path, split, scene_name)
    os.makedirs(save_path, exist_ok=True)



    np.save(os.path.join(save_path, "coord.npy"), pos.astype(np.float32))
    np.save(os.path.join(save_path, "color.npy"), intensity.astype(np.float32))
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

def parse_lidar(dataset_root, grid_size):
    print("Reading lidar files...")
    classes = [
        'ground',
        'vegetaion',
        'buildings',
        'cars'
    ]
    class2label = {cls: i for i, cls in enumerate(classes)}

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
        root_path 是 raw 文件夹的路径
        - tile
        - processed
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
            if laz_file.endswith(".laz"):  # 确保只处理 .laz 文件
                print('start', laz_file)
                laz_file_path = os.path.join(folder_path, laz_file)
                # 调用处理函数，并传入 split
                execuate_las(laz_file_path, grid_size, folder, save_path)

def main_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", help="Path where raw datasets are located.", default=r'/workspace/data/'
    )
    

    parser.add_argument(
        "--num_workers", default=1, type=int, help="Num workers for preprocessing."
    )
    parser.add_argument(
        "--grid_size", default=1.0, type=float, help="grid size in meters."
    )
    args = parser.parse_args()

    print("Loading LAS information ...")

    parse_lidar(args.dataset_root, args.grid_size)

if __name__ == "__main__":
    # main_process()
    main_preprocess()

