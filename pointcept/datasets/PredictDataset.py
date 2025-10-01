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
                 grid_size=0.1,
                 save_blocks_to_ply=False,
                 ply_save_dir="blocks_ply", **kwargs):
        self.laz_file = laz_file
        self.has_ground_truth = has_ground_truth
        self.num_points_per_block = num_points_per_block
        self.overlap_ratio = overlap_ratio
        self.grid_size = grid_size
        self.num_total_points = 0
        self.vote_rounds = kwargs.get('vote_rounds', 3)  # 投票轮数
        self.loaded_data = None  # 缓存已加载的数据
        self.processed_coord = None  # 经过transform后的坐标数据，用于分块
        self.save_blocks_to_ply = save_blocks_to_ply  # 是否保存块为PLY格式
        self.ply_save_dir = ply_save_dir  # PLY文件保存目录

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

        # 创建PLY保存目录
        if self.save_blocks_to_ply:
            self._create_ply_save_dir()

        # 加载LAZ文件数据（如果是单文件模式）
        if self.single_laz_mode and self.laz_file and os.path.exists(self.laz_file):
            self._load_laz_file(self.laz_file)
            # 缓存处理后的数据
            self._cache_loaded_data()
            # 为测试模式创建空间分块（在数据处理后）
            if test_mode:
                self._create_spatial_blocks_with_kdtree()

    def _create_ply_save_dir(self):
        """创建PLY文件保存目录"""
        if self.single_laz_mode and self.laz_file:
            # 单文件模式下，在文件所在目录创建子目录
            file_dir = os.path.dirname(self.laz_file)
            self.ply_save_dir = os.path.join(file_dir, self.ply_save_dir)
        else:
            # 多文件模式下，在数据根目录创建目录
            self.ply_save_dir = os.path.join(self.data_root, self.ply_save_dir)

        os.makedirs(self.ply_save_dir, exist_ok=True)
        print(f"PLY files will be saved to: {self.ply_save_dir}")

    def save_block_to_ply(self, block_idx, coordinates, filename=None):
        """
        将块保存为PLY格式文件
        :param block_idx: 块索引
        :param coordinates: 三维坐标数据 (N, 3)
        :param filename: 自定义文件名，如果为None则自动生成
        """
        if not self.save_blocks_to_ply:
            return

        # 确保坐标是numpy数组
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()

        # 确保坐标是3D的
        if coordinates.ndim == 2 and coordinates.shape[1] == 3:
            pass
        elif coordinates.ndim == 1:
            coordinates = coordinates.reshape(-1, 3)
        else:
            print(f"Warning: Invalid coordinate shape {coordinates.shape} for block {block_idx}")
            return

        # 生成文件名
        if filename is None:
            if self.single_laz_mode and self.laz_file:
                base_name = os.path.splitext(os.path.basename(self.laz_file))[0]
                filename = f"{base_name}_block_{block_idx:04d}.ply"
            else:
                filename = f"block_{block_idx:04d}.ply"

        ply_path = os.path.join(self.ply_save_dir, filename)

        # 保存为PLY格式
        try:
            with open(ply_path, 'w') as f:
                # PLY文件头
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(coordinates)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")

                # 写入坐标数据
                for coord in coordinates:
                    f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")

            print(f"Saved block {block_idx} to PLY file: {ply_path}")
        except Exception as e:
            print(f"Error saving block {block_idx} to PLY: {str(e)}")

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

    def _cache_loaded_data(self):
        """缓存已加载的数据"""
        if self.single_laz_mode and self.current_coord is not None:
            # 构建数据字典
            data_dict = {}
            data_dict["coord"] = self.current_coord.copy()
            data_dict["color"] = self.current_color.copy()
            data_dict["name"] = os.path.basename(self.laz_file)
            data_dict["split"] = self.split
            data_dict["offset"] = np.array([self.current_coord.shape[0]], dtype=np.int32)

            if self.current_segment is not None:
                data_dict["segment"] = self.current_segment.copy()
            else:
                data_dict["segment"] = np.ones(self.current_coord.shape[0], dtype=np.int32) * -1

            # 应用变换
            if self.transform is not None:
                print(f"Applying transform to data (size: {data_dict['coord'].shape[0]})")
                transformed_data = self.transform(data_dict)

                # 处理transform返回的可能格式
                if isinstance(transformed_data, list):
                    print(f"Warning: Transform returned list with {len(transformed_data)} elements")
                    # 如果是列表，取第一个元素（假设是主要数据）
                    if len(transformed_data) > 0:
                        data_dict = transformed_data[0]
                    else:
                        print("Error: Transform returned empty list")
                elif isinstance(transformed_data, dict):
                    data_dict = transformed_data
                else:
                    print(f"Warning: Transform returned unexpected type {type(transformed_data)}")

                print(f"Data size after transform: {data_dict['coord'].shape[0]}")

            # 确保数据是预期的格式
            if not isinstance(data_dict, dict):
                raise RuntimeError(f"Expected data_dict to be dict, got {type(data_dict)}")

            if "coord" not in data_dict:
                raise RuntimeError("Data dict missing 'coord' key after transform")

            # 保存处理后的坐标用于分块
            if isinstance(data_dict["coord"], torch.Tensor):
                self.processed_coord = data_dict["coord"].cpu().numpy()
            else:
                self.processed_coord = data_dict["coord"].copy()

            self.loaded_data = data_dict
            print(f"Cached loaded data for single file mode")

    def get_data(self, idx):
        """获取数据（重写DefaultDataset的方法）"""
        if self.cache:
            cache_name = f"pointcept-{self.get_data_name(idx)}"
            cached_data = shared_dict(cache_name)
            if cached_data is not None:
                return cached_data

        # 单文件模式下直接使用缓存的数据
        if self.single_laz_mode and self.loaded_data is not None:
            return copy.deepcopy(self.loaded_data)

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

    def _create_spatial_blocks_with_kdtree(self):
        """使用KDTree创建空间分块 - 简化版本"""
        print(f"Creating spatial blocks with KDTree, {self.num_points_per_block} points per block")

        if self.processed_coord is None:
            if self.current_coord is not None:
                self.processed_coord = self.current_coord
                print("Warning: Using original coordinates for blocking (no processed coordinates available)")
            else:
                raise RuntimeError("No processed coordinate data available for blocking")

        num_points = self.processed_coord.shape[0]
        self.current_blocks = []

        if num_points == 0:
            print("No points to process")
            return

        # 如果点数少于每个块的点数，直接返回整个点云作为一个块
        if num_points <= self.num_points_per_block:
            self.current_blocks.append(np.arange(num_points, dtype=np.int32))
            print(f"Only {num_points} points, creating single block")

            # 如果需要，保存块为PLY格式
            if self.save_blocks_to_ply:
                block_coords = self.processed_coord[:]
                self.save_block_to_ply(0, block_coords)

            return

        print(f"Building KDTree for {num_points} points...")

        # 创建KDTree
        kdtree = KDTree(self.processed_coord, leaf_size=40)

        # 计算需要的块数
        # 考虑重叠，每个块实际覆盖的新点为 num_points_per_block * (1 - overlap_ratio)
        effective_points_per_block = int(self.num_points_per_block * (1 - self.overlap_ratio))
        num_blocks = (num_points + effective_points_per_block - 1) // effective_points_per_block
        print(f"Estimated {num_blocks} blocks needed (considering {self.overlap_ratio * 100}% overlap)")

        # 为了均匀覆盖，在点云中均匀选择查询点
        step = max(1, num_points // num_blocks)
        query_indices = np.arange(0, num_points, step)[:num_blocks]

        print(f"Selected {len(query_indices)} query points")

        # 为每个查询点创建块
        for block_id, query_idx in enumerate(query_indices):
            # 获取查询点坐标
            query_point = self.processed_coord[query_idx]

            # 使用KDTree搜索最近的num_points_per_block个点
            k = min(self.num_points_per_block, num_points)
            distances, indices = kdtree.query([query_point], k=k)
            block_indices = indices[0]

            # 确保块索引有效
            block_indices = block_indices[block_indices < num_points]

            # 添加到块列表
            self.current_blocks.append(block_indices.astype(np.int32))

            # 如果需要，保存块为PLY格式
            if self.save_blocks_to_ply:
                block_coords = self.processed_coord[block_indices]
                self.save_block_to_ply(block_id, block_coords)

            # 打印进度
            if (block_id + 1) % 10 == 0 or (block_id + 1) == len(query_indices):
                print(f"Created {block_id + 1}/{len(query_indices)} blocks")

        # 检查是否有遗漏的点（可选）
        all_indices = set()
        for block in self.current_blocks:
            all_indices.update(block)

        print(f"Created {len(self.current_blocks)} blocks")
        print(f"Total points in blocks: {sum(len(block) for block in self.current_blocks)}")
        print(f"Unique points covered: {len(all_indices)} / {num_points}")

        # 验证块大小
        block_sizes = [len(block) for block in self.current_blocks]
        print(
            f"Block size distribution: min={min(block_sizes)}, max={max(block_sizes)}, avg={np.mean(block_sizes):.1f}")

    def prepare_test_data(self, idx):
        """准备测试数据（简化版本）"""
        if self.test_mode and hasattr(self, 'current_blocks') and len(self.current_blocks) > 0:
            # 使用预创建的块
            block_idx = idx % len(self.current_blocks)
            block_indices = self.current_blocks[block_idx].copy()

            print(f"Processing block {block_idx}: {len(block_indices)} points")

            # 单文件模式下直接使用缓存的数据
            if self.single_laz_mode and self.loaded_data is not None:
                full_data = self.loaded_data
            else:
                # 获取完整数据
                full_data = self.get_data(idx // len(self.current_blocks))

        else:
            # 默认测试模式
            return super().prepare_test_data(idx)

        # 提取块数据
        data_dict = {}
        # 确保coord是torch张量
        if isinstance(full_data["coord"], torch.Tensor):
            data_dict["coord"] = full_data["coord"][block_indices].clone()
            data_dict["grid_coord"] = full_data["grid_coord"][block_indices].clone()
            data_dict["inverse"] = full_data["inverse"][block_indices].clone()
            data_dict["feat"] = full_data["feat"][block_indices].clone()
        elif isinstance(full_data["coord"], list):
            # 处理列表格式
            if len(full_data["coord"]) > 0:
                coord_data = full_data["coord"][0]
                if isinstance(coord_data, torch.Tensor):
                    data_dict["coord"] = coord_data[block_indices].clone()
                else:
                    data_dict["coord"] = torch.tensor(coord_data[block_indices].copy())
            else:
                raise RuntimeError("full_data['coord'] is empty list")
        else:
            data_dict["coord"] = torch.tensor(full_data["coord"][block_indices].copy())

        # 确保index是torch张量
        data_dict["index"] = torch.from_numpy(block_indices.copy())
        data_dict["offset"] = torch.tensor([len(block_indices)], dtype=torch.int32)

        if "segment" in full_data:
            if isinstance(full_data["segment"], torch.Tensor):
                data_dict["segment"] = full_data["segment"][block_indices].clone()
            elif isinstance(full_data["segment"], list):
                if len(full_data["segment"]) > 0:
                    segment_data = full_data["segment"][0]
                    if isinstance(segment_data, torch.Tensor):
                        data_dict["segment"] = segment_data[block_indices].clone()
                    else:
                        data_dict["segment"] = torch.tensor(segment_data[block_indices].copy())
                else:
                    data_dict["segment"] = torch.ones(len(block_indices), dtype=torch.int32) * -1
            else:
                data_dict["segment"] = torch.tensor(full_data["segment"][block_indices].copy())
        else:
            data_dict["segment"] = torch.ones(len(block_indices), dtype=torch.int32) * -1

        # 投票模式下，直接返回数据字典，不生成fragment_list
        result_dict = {
            "data": data_dict,
            "segment": data_dict.pop("segment", None),
            "block_idx": block_idx,
            "num_blocks": len(self.current_blocks),
            "vote_rounds": self.vote_rounds
        }

        return result_dict

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

        pipeline |= pdal.Writer.las(filename=output_path, **las_kwargs)
        pipeline.execute()

        print("Successfully saved LAZ file with predictions")