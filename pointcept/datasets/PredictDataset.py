import json
from uuid import uuid4
import os
import time
import numpy as np
from collections import OrderedDict, abc
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

try:
    import pointops
except:
    pointops = None
from .defaults import DefaultDataset
from .builder import DATASETS
from pointcept.utils.cache import shared_dict
import os
import numpy as np
import pdal
from sklearn.neighbors import KDTree

import glob
import json
import copy

@DATASETS.register_module()
class LAZDataset(DefaultDataset):
    """
    LAZ文件数据集类
    支持直接读取LAZ文件，无需格式转换
    继承自DefaultDataset以保持与Pointcept框架的兼容性
    """

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
        "intensity",
        "index"
    ]

    def __init__(self,
                 laz_file=None,
                 split="test",
                 data_root="data/dataset",
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 cache=False,
                 ignore_index=-1,
                 loop=1,
                 has_ground_truth=True,
                 num_points_per_block=60000,
                 overlap_ratio=0.1,
                 grid_size=0.1,
                 **kwargs):
        self.laz_file = laz_file
        self.has_ground_truth = has_ground_truth
        self.num_points_per_block = num_points_per_block
        self.overlap_ratio = overlap_ratio
        self.grid_size = grid_size

        # 如果提供了laz_file，直接使用它
        if laz_file is not None:
            self.single_laz_mode = True
            self.data_root = os.path.dirname(laz_file)
            self.split = os.path.basename(laz_file)
        else:
            self.single_laz_mode = False

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            cache=cache,
            ignore_index=ignore_index,
            loop=loop, **kwargs
        )

        # 加载LAZ文件数据（如果是单文件模式）
        if self.single_laz_mode and self.laz_file and os.path.exists(self.laz_file):
            self._load_laz_file(self.laz_file)
            # 为测试模式创建空间分块
            if test_mode:
                self._create_spatial_blocks()

    def get_data_list(self):
        """获取数据列表"""
        if self.single_laz_mode and self.laz_file:
            # 单LAZ文件模式
            return [self.laz_file]
        else:
            # 多文件模式，查找所有LAZ文件
            if isinstance(self.split, str):
                split_list = [self.split]
            elif isinstance(self.split, abc.Sequence):
                split_list = self.split
            else:
                raise NotImplementedError

            data_list = []
            for split in split_list:
                # 查找LAZ文件
                laz_files = glob.glob(os.path.join(self.data_root, split, "*.laz"))
                laz_files += glob.glob(os.path.join(self.data_root, split, "*.las"))
                data_list += laz_files

                # 如果是文件列表，读取文件内容
                if os.path.isfile(os.path.join(self.data_root, split)):
                    with open(os.path.join(self.data_root, split)) as f:
                        file_list = json.load(f)
                        for data in file_list:
                            if data.endswith(".laz") or data.endswith(".las"):
                                data_list.append(os.path.join(self.data_root, data))

            return data_list

    def _load_laz_file(self, file_path):
        """加载LAZ文件"""
        print(f"Loading LAZ file: {file_path}")

        # PDAL管道
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=file_path)

        # 统计信息
        stats_dimensions = "Intensity,Red,Blue,Green"
        if self.has_ground_truth:
            pipeline |= pdal.Filter.range(limits="Classification[2:6], Classification[8:8]")
            pipeline |= pdal.Filter.assign(value=[
                f"Classification = 0 WHERE Classification == 2",
                f"Classification = 1 WHERE ((Classification >= 3) && (Classification <= 5))",
                f"Classification = 2 WHERE Classification == 6",
                f"Classification = 3 WHERE Classification == 8"
            ])
        else:
            print("No label, only process point cloud.")


        pipeline |= pdal.Filter.stats(dimensions=stats_dimensions)

        # 体素下采样
        pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=self.grid_size)
        pipeline.execute()

        # 获取元数据和点云数据
        metadata = pipeline.metadata['metadata']
        arrays = pipeline.arrays[0]

        # 基本信息
        self.current_metadata = metadata
        self.current_arrays = arrays

        # 坐标偏移
        minx = metadata['readers.las']['minx']
        miny = metadata['readers.las']['miny']
        minz = metadata['readers.las']['minz']
        self.current_offset = np.array([minx, miny, minz], dtype=np.float32)

        # 提取点云数据
        self.current_coord = np.concatenate([
            np.expand_dims(arrays['X'] - minx, 1),
            np.expand_dims(arrays['Y'] - miny, 1),
            np.expand_dims(arrays['Z'] - minz, 1)
        ], axis=-1).astype(np.float32)

        # 提取颜色信息
        self.current_color = np.concatenate([
            np.expand_dims(arrays['Red'] / 255.0, 1),
            np.expand_dims(arrays['Green'] / 255.0, 1),
            np.expand_dims(arrays['Blue'] / 255.0, 1)
        ], axis=-1).astype(np.float32)

        # 提取标签（如果有）
        self.current_segment = None
        if self.has_ground_truth:
            self.current_segment = arrays['Classification'].astype(np.int32)
            # 处理忽略标签
            self.current_segment[self.current_segment == self.ignore_index] = -1

        print(f"Loaded {self.current_coord.shape[0]} points from LAZ file")

    def _create_spatial_blocks(self):
        """创建空间分块"""
        print(f"Creating spatial blocks with {self.num_points_per_block} points per block")

        if not hasattr(self, 'current_coord'):
            raise RuntimeError("LAZ file not loaded properly")

        # 创建KDTree
        kdtree = KDTree(self.current_coord[:, :2])  # 使用2D坐标进行分块

        num_points = self.current_coord.shape[0]
        self.current_blocks = []
        processed = np.zeros(num_points, dtype=bool)
        block_size = int(self.num_points_per_block * (1 + self.overlap_ratio))

        while np.sum(processed) < num_points:
            # 找到未处理的点
            unprocessed = np.where(~processed)[0]
            if len(unprocessed) == 0:
                break

            # 选择块中心
            center_idx = unprocessed[0]
            center_pos = self.current_coord[center_idx: center_idx + 1, :2]

            # 查询最近的点
            distances, indices = kdtree.query(center_pos, k=min(block_size, num_points))
            indices = indices[0]

            # 过滤已处理的点
            new_indices = indices[~processed[indices]]
            if len(new_indices) < 100:
                new_indices = indices[:min(100, len(indices))]

            # 添加块
            self.current_blocks.append(new_indices)

            # 标记为已处理
            processed[new_indices[:int(len(new_indices) * (1 - self.overlap_ratio))]] = True

        print(f"Created {len(self.current_blocks)} spatial blocks")

    def get_data(self, idx):
        """获取数据（重写DefaultDataset的方法）"""
        if self.cache:
            cache_name = f"pointcept-{self.get_data_name(idx)}"
            cached_data = shared_dict(cache_name)
            if cached_data is not None:
                return cached_data

        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        # 加载LAZ文件
        self._load_laz_file(data_path)

        # 构建数据字典
        data_dict = {}
        data_dict["coord"] = self.current_coord
        data_dict["color"] = self.current_color
        data_dict["name"] = name
        data_dict["split"] = split
        data_dict["offset"] = np.array([self.current_coord.shape[0]], dtype=np.int32)

        if self.current_segment is not None:
            data_dict["segment"] = self.current_segment
        else:
            data_dict["segment"] = np.ones(self.current_coord.shape[0], dtype=np.int32) * -1

        # 添加原始坐标信息（用于后续保存）
        # data_dict["original_metadata"] = self.current_metadata
        # data_dict["original_arrays"] = self.current_arrays
        # data_dict["original_offset"] = self.current_offset

        return data_dict

    def prepare_test_data(self, idx):
        """准备测试数据（重写DefaultDataset的方法）"""
        if self.test_mode and hasattr(self, 'current_blocks'):
            # 使用预创建的块
            block_idx = idx % len(self.current_blocks)
            block_indices = self.current_blocks[block_idx]

            # 获取完整数据
            full_data = self.get_data(idx // len(self.current_blocks))

            # 提取块数据
            data_dict = {}
            data_dict["coord"] = full_data["coord"][block_indices].copy()
            data_dict["color"] = full_data["color"][block_indices].copy()
            data_dict["index"] = block_indices.copy()  # 保存原始索引
            data_dict["name"] = full_data["name"]
            data_dict["split"] = full_data["split"]

            if "segment" in full_data:
                data_dict["segment"] = full_data["segment"][block_indices].copy()
            else:
                data_dict["segment"] = np.ones(len(block_indices), dtype=np.int32) * -1

            # 应用变换
            data_dict = self.transform(data_dict)

            result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
            if "origin_segment" in data_dict:
                result_dict["origin_segment"] = data_dict.pop("origin_segment")
                result_dict["inverse"] = data_dict.pop("inverse")

            # 测试时的数据增强和预处理
            data_dict_list = []
            for aug in self.aug_transform:
                data_dict_list.append(aug(copy.deepcopy(data_dict)))

            fragment_list = []
            for data in data_dict_list:
                if self.test_voxelize is not None:
                    data_part_list = self.test_voxelize(data)
                else:
                    data["index"] = np.arange(data["coord"].shape[0])
                    data_part_list = [data]
                for data_part in data_part_list:
                    if self.test_crop is not None:
                        data_part = self.test_crop(data_part)
                    else:
                        data_part = [data_part]
                    fragment_list += data_part

            for i in range(len(fragment_list)):
                fragment_list[i] = self.post_transform(fragment_list[i])

            result_dict["fragment_list"] = fragment_list
            result_dict["block_idx"] = block_idx
            result_dict["num_blocks"] = len(self.current_blocks)

            return result_dict
        else:
            # 默认测试模式
            return super().prepare_test_data(idx)

    def __getitem__(self, idx):
        """获取数据项"""
        if self.test_mode and hasattr(self, 'current_blocks'):
            # 测试模式下，每个块作为一个数据项
            return self.prepare_test_data(idx)
        else:
            return super().__getitem__(idx)

    def __len__(self):
        """数据集长度"""
        if self.test_mode and hasattr(self, 'current_blocks'):
            # 测试模式下，长度为块的数量
            return len(self.current_blocks) * self.loop
        else:
            return super().__len__()

    def save_predictions(self, predictions, output_path):
        """保存预测结果为LAZ文件"""
        print(f"Saving predictions to LAZ file: {output_path}")

        if not hasattr(self, 'current_arrays'):
            raise RuntimeError("No LAZ data loaded.")

        # 创建新的PDAL数组
        new_array = self.current_arrays.copy()

        # 更新分类标签
        new_array['Classification'] = predictions.astype(np.int32)

        # 创建PDAL管道
        pipeline = pdal.Pipeline(arrays=[new_array])

        # 设置LAZ文件参数
        las_kwargs = {
            'minor_version': 4,
            'scale_x': 0.001,
            'scale_y': 0.001,
            'scale_z': 0.001,
            'offset_x': 'auto',
            'offset_y': 'auto',
            'offset_z': 'auto'
        }

        # 添加空间参考
        if hasattr(self, 'current_metadata'):
            srs = self.current_metadata['readers.las'].get('comp_spatialreference', '')
            if srs:
                las_kwargs['a_srs'] = srs

        pipeline |= pdal.Writer.las(filename=output_path, **las_kwargs)
        pipeline.execute()

        print("Successfully saved LAZ file with predictions")