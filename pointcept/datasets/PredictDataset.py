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

import pointops

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
class LAZDatasetVote(DefaultDataset):
    """
    LAZ文件数据集类（投票版本）
    支持直接读取LAZ文件，无需格式转换
    去除fragment_list操作，支持投票机制
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
        "index",
        "offset"
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
                 grid_size=0.1,** kwargs):
        self.laz_file = laz_file
        self.has_ground_truth = has_ground_truth
        self.num_points_per_block = num_points_per_block
        self.overlap_ratio = overlap_ratio
        self.grid_size = grid_size
        self.num_total_points = 0
        self.vote_rounds = kwargs.get('vote_rounds', 3)  # 投票轮数

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

        # 坐标偏移（用于还原原始坐标）
        minx = metadata['readers.las']['minx']
        miny = metadata['readers.las']['miny']
        minz = metadata['readers.las']['minz']
        self.current_coord_offset = np.array([minx, miny, minz], dtype=np.float32)

        # 提取点云数据（相对于偏移的坐标）
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

        self.num_total_points = self.current_coord.shape[0]
        print(f"Loaded {self.num_total_points} points from LAZ file")

    def _create_spatial_blocks(self):
        """创建空间分块 - 优化版本"""
        print(f"Creating spatial blocks with {self.num_points_per_block} points per block")

        if not hasattr(self, 'current_coord'):
            raise RuntimeError("LAZ file not loaded properly")

        num_points = self.current_coord.shape[0]
        self.current_blocks = []

        if num_points == 0:
            print("No points to process")
            return

        # 使用网格分块代替KDTree，效率更高
        print("Using grid-based spatial partitioning for better performance")

        # 获取坐标范围
        x_coords = self.current_coord[:, 0]
        y_coords = self.current_coord[:, 1]

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        # 计算块的空间大小
        # 根据每块点数和点云密度估算网格大小
        point_density = num_points / ((max_x - min_x) * (max_y - min_y))
        block_area = self.num_points_per_block / point_density
        block_size_xy = np.sqrt(block_area) * (1 + self.overlap_ratio)

        print(f"Point density: {point_density:.2f} points/m²")
        print(f"Estimated block size: {block_size_xy:.2f}m x {block_size_xy:.2f}m")

        # 创建网格
        grid_x = np.arange(min_x, max_x + block_size_xy, block_size_xy)
        grid_y = np.arange(min_y, max_y + block_size_xy, block_size_xy)

        print(f"Creating grid with {len(grid_x)} x {len(grid_y)} cells")

        # 为每个点分配网格索引
        point_grid_x = np.digitize(x_coords, grid_x) - 1
        point_grid_y = np.digitize(y_coords, grid_y) - 1

        # 确保网格索引在有效范围内
        point_grid_x = np.clip(point_grid_x, 0, len(grid_x) - 2)
        point_grid_y = np.clip(point_grid_y, 0, len(grid_y) - 2)

        # 计算每个网格单元的点
        grid_dict = {}
        for i in range(num_points):
            gx, gy = point_grid_x[i], point_grid_y[i]
            key = (gx, gy)
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)

        print(f"Found {len(grid_dict)} non-empty grid cells")

        # 处理每个网格单元
        processed = set()
        block_id = 0

        for (gx, gy), point_indices in grid_dict.items():
            # 跳过已处理的点
            unprocessed_indices = [idx for idx in point_indices if idx not in processed]
            if not unprocessed_indices:
                continue

            # 如果当前网格单元的点太少，合并相邻网格
            if len(unprocessed_indices) < self.num_points_per_block * 0.5:
                # 合并相邻的4个网格
                neighboring_keys = [
                    (gx, gy), (gx + 1, gy), (gx, gy + 1), (gx + 1, gy + 1)
                ]
                combined_indices = []
                for key in neighboring_keys:
                    if key in grid_dict:
                        combined_indices.extend([idx for idx in grid_dict[key] if idx not in processed])

                if len(combined_indices) > 0:
                    unprocessed_indices = combined_indices

            # 将点分成多个块（如果需要）
            num_full_blocks = len(unprocessed_indices) // self.num_points_per_block
            remaining_points = len(unprocessed_indices) % self.num_points_per_block

            # 创建完整的块
            for i in range(num_full_blocks):
                start_idx = i * self.num_points_per_block
                end_idx = start_idx + self.num_points_per_block
                block_indices = unprocessed_indices[start_idx:end_idx]

                self.current_blocks.append(np.array(block_indices, dtype=np.int32))

                # 标记为已处理（减去重叠部分）
                num_to_mark = int(len(block_indices) * (1 - self.overlap_ratio))
                for idx in block_indices[:num_to_mark]:
                    processed.add(idx)

                block_id += 1

            # 创建剩余的块
            if remaining_points > 100:  # 只保留点数足够的块
                start_idx = num_full_blocks * self.num_points_per_block
                block_indices = unprocessed_indices[start_idx:]

                self.current_blocks.append(np.array(block_indices, dtype=np.int32))

                # 标记为已处理（减去重叠部分）
                num_to_mark = int(len(block_indices) * (1 - self.overlap_ratio))
                for idx in block_indices[:num_to_mark]:
                    processed.add(idx)

                block_id += 1

        # 处理剩余的孤立点
        remaining_points = [idx for idx in range(num_points) if idx not in processed]
        if remaining_points:
            # 将剩余点分配到现有块中或创建新块
            if len(remaining_points) <= self.num_points_per_block:
                self.current_blocks.append(np.array(remaining_points, dtype=np.int32))
            else:
                # 分成多个小块
                for i in range(0, len(remaining_points), self.num_points_per_block):
                    end_idx = min(i + self.num_points_per_block, len(remaining_points))
                    self.current_blocks.append(np.array(remaining_points[i:end_idx], dtype=np.int32))

        print(f"Created {len(self.current_blocks)} spatial blocks")
        print(f"Total points processed: {sum(len(block) for block in self.current_blocks)}")

        # 验证所有点都被处理
        all_indices = set()
        for block in self.current_blocks:
            all_indices.update(block)
        print(f"Unique points in blocks: {len(all_indices)} / {num_points}")

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


        return data_dict

    def prepare_test_data(self, idx):
        """准备测试数据（简化版本，去除fragment_list）"""
        if self.test_mode and hasattr(self, 'current_blocks') and len(self.current_blocks) > 0:
            # 使用预创建的块
            block_idx = idx % len(self.current_blocks)
            block_indices = self.current_blocks[block_idx].copy()

            # 确保索引在有效范围内
            num_total_points = self.current_coord.shape[0]
            block_indices = block_indices[block_indices < num_total_points]

            if len(block_indices) == 0:
                raise RuntimeError(f"Block {block_idx} has no valid points")

            # 获取完整数据
            full_data = self.get_data(idx // len(self.current_blocks))
            full_data = self.transform(full_data)
            # 提取块数据
            data_dict = {}
            data_dict["coord"] = full_data["coord"][block_indices].clone()
            data_dict["index"] = torch.from_numpy(block_indices).clone()
            # data_dict["name"] = full_data["name"]
            # data_dict["split"] = full_data["split"]
            data_dict["offset"] = torch.tensor([len(block_indices)], dtype=torch.int32)

            if "segment" in full_data:
                data_dict["segment"] = full_data["segment"][block_indices].clone()
            else:
                data_dict["segment"] = torch.ones(len(block_indices), dtype=torch.int32) * -1

            # 应用变换
            # data_dict = self.transform(data_dict)

            # 投票模式下，直接返回数据字典，不生成fragment_list
            result_dict = {
                "data": data_dict,
                "segment": data_dict.pop("segment", None),
                # "name": data_dict.pop("name", f"block_{block_idx}"),
                "block_idx": block_idx,
                "num_blocks": len(self.current_blocks),
                "vote_rounds": self.vote_rounds
            }

            return result_dict
        else:
            # 默认测试模式
            return super().prepare_test_data(idx)

    def __getitem__(self, idx):
        """获取数据项"""
        if self.test_mode and hasattr(self, 'current_blocks') and len(self.current_blocks) > 0:
            # 测试模式下，每个块作为一个数据项
            return self.prepare_test_data(idx)
        else:
            return super().__getitem__(idx)

    def __len__(self):
        """数据集长度"""
        if self.test_mode and hasattr(self, 'current_blocks') and len(self.current_blocks) > 0:
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

        pipeline |= pdal.Writer.las(filename=output_path,** las_kwargs)
        pipeline.execute()

        print("Successfully saved LAZ file with predictions")