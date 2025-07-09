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
from .preprocess_farmland import parse_lidar


def las_tile(raw_las_path, tile_size):
    # tile las file using pdal

    path_prefix = os.path.splitext(raw_las_path)[0]
    

    # 构建 PDAL 管道配置
    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": f"{raw_las_path}"
            },
            {
                "type": "filters.splitter",
                "edge_length": tile_size  # 每个 tile 的边长（单位：米）
            },
            {
                "type": "writers.las",
                "filename": f"{path_prefix}#.laz"  # 使用 # 号表示索引
            }
        ]
    }

    # 创建并执行管道
    pipeline = pdal.Pipeline(pipeline_json)
    count = pipeline.execute()

    print(f"Processed {count} points.")

def process_lidar_files(dataset_root, tile_size):
    print("Processing LIDAR files...")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_root, 'raw', split)
        for laz_file in os.listdir(split_dir):
            if laz_file.endswith(".laz"):
                laz_file_path = os.path.join(split_dir, laz_file)
                las_tile(laz_file_path, tile_size)

def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", help="Path where raw datasets are located.", default=r''
    )
    parser.add_argument(
        "--output_root",

        help="Output path where area folders will be located.", default=r''
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Num workers for preprocessing."
    )
    parser.add_argument(
        "--grid_size", default=0.1, type=float, help="grid size in meters."
    )
    args = parser.parse_args()

    print("Tiling las files ...")

    

    parse_lidar(args.dataset_root, args.grid_size)
