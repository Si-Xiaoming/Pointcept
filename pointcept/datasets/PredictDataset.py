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
import pdal
from sklearn.neighbors import KDTree
import glob
import copy


@DATASETS.register_module()
class LAZDatasetVote(DefaultDataset):
    """
    LAZ文件数据集类（投票版本）- 修复版
    关键修复：正确处理transform前后的索引映射关系
    """

    VALID_ASSETS = [
        "coord", "color", "normal", "strength", "segment",
        "instance", "pose", "intensity", "index", "offset"
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
                 save_blocks_to_ply=True,
                 ply_save_dir="blocks_ply", **kwargs):

        self.laz_file = laz_file
        self.has_ground_truth = has_ground_truth
        self.num_points_per_block = num_points_per_block
        self.overlap_ratio = overlap_ratio
        self.grid_size = grid_size
        self.num_total_points = 0
        self.vote_rounds = kwargs.get('vote_rounds', 3)

        # 数据缓存
        self.loaded_data = None
        self.processed_coord = None

        # **关键修复：添加原始数据和映射关系**
        self.original_coord = None  # 原始坐标（transform之前）
        self.original_segment = None  # 原始标签
        self.transform_mapping = None  # transform前后的索引映射 [processed_idx] -> original_idx

        self.save_blocks_to_ply = save_blocks_to_ply
        self.ply_save_dir = ply_save_dir

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

        if self.save_blocks_to_ply:
            self._create_ply_save_dir()

        if self.single_laz_mode and self.laz_file and os.path.exists(self.laz_file):
            self._load_laz_file(self.laz_file)
            self._cache_loaded_data()
            if test_mode:
                self._create_spatial_blocks_with_kdtree()

    def _create_ply_save_dir(self):
        """创建PLY文件保存目录"""
        if self.single_laz_mode and self.laz_file:
            file_dir = os.path.dirname(self.laz_file)
            self.ply_save_dir = os.path.join(file_dir, self.ply_save_dir)
        else:
            self.ply_save_dir = os.path.join(self.data_root, self.ply_save_dir)

        os.makedirs(self.ply_save_dir, exist_ok=True)
        print(f"PLY files will be saved to: {self.ply_save_dir}")

    def get_data_list(self):
        """获取数据列表"""
        if self.single_laz_mode and self.laz_file:
            return [self.laz_file]
        else:
            if isinstance(self.split, str):
                split_list = [self.split]
            elif isinstance(self.split, abc.Sequence):
                split_list = self.split
            else:
                raise NotImplementedError

            data_list = []
            for split in split_list:
                laz_files = glob.glob(os.path.join(self.data_root, split, "*.laz"))
                laz_files += glob.glob(os.path.join(self.data_root, split, "*.las"))
                data_list += laz_files

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

        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=file_path)

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
        pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=self.grid_size)
        pipeline.execute()

        metadata = pipeline.metadata['metadata']
        arrays = pipeline.arrays[0]

        self.current_metadata = metadata
        self.current_arrays = arrays

        minx = metadata['readers.las']['minx']
        miny = metadata['readers.las']['miny']
        minz = metadata['readers.las']['minz']
        self.current_coord_offset = np.array([minx, miny, minz], dtype=np.float32)

        self.current_coord = np.concatenate([
            np.expand_dims(arrays['X'] - minx, 1),
            np.expand_dims(arrays['Y'] - miny, 1),
            np.expand_dims(arrays['Z'] - minz, 1)
        ], axis=-1).astype(np.float32)

        self.current_color = np.concatenate([
            np.expand_dims(arrays['Red'] / 255.0, 1),
            np.expand_dims(arrays['Green'] / 255.0, 1),
            np.expand_dims(arrays['Blue'] / 255.0, 1)
        ], axis=-1).astype(np.float32)

        self.current_segment = None
        if self.has_ground_truth:
            self.current_segment = arrays['Classification'].astype(np.int32)
            self.current_segment[self.current_segment == self.ignore_index] = -1

        # **保存原始数据（transform之前）**
        self.original_coord = self.current_coord.copy()
        if self.current_segment is not None:
            self.original_segment = self.current_segment.copy()

        self.num_total_points = self.current_coord.shape[0]
        print(f"Loaded {self.num_total_points} points from LAZ file (original size)")

    def _cache_loaded_data(self):
        """缓存已加载的数据并建立映射关系"""
        if self.single_laz_mode and self.current_coord is not None:
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

            # **关键修复：添加原始索引，用于追踪transform的影响**
            data_dict["original_index"] = np.arange(self.current_coord.shape[0], dtype=np.int32)

            if self.transform is not None:
                print(f"Applying transform to data (size: {data_dict['coord'].shape[0]})")
                transformed_data = self.transform(data_dict)

                if isinstance(transformed_data, list):
                    print(f"Warning: Transform returned list with {len(transformed_data)} elements")
                    if len(transformed_data) > 0:
                        data_dict = transformed_data[0]
                    else:
                        raise RuntimeError("Transform returned empty list")
                elif isinstance(transformed_data, dict):
                    data_dict = transformed_data
                else:
                    raise RuntimeError(f"Transform returned unexpected type {type(transformed_data)}")

                print(f"Data size after transform: {data_dict['coord'].shape[0]}")

                # **建立映射关系**
                if "original_index" in data_dict:
                    if isinstance(data_dict["original_index"], torch.Tensor):
                        self.transform_mapping = data_dict["original_index"].cpu().numpy()
                    else:
                        self.transform_mapping = data_dict["original_index"]
                    print(
                        f"Transform mapping created: {len(self.transform_mapping)} processed points -> original indices")
                else:
                    print("Warning: No original_index in transformed data, creating identity mapping")
                    self.transform_mapping = np.arange(data_dict['coord'].shape[0], dtype=np.int32)

            if not isinstance(data_dict, dict):
                raise RuntimeError(f"Expected data_dict to be dict, got {type(data_dict)}")

            if "coord" not in data_dict:
                raise RuntimeError("Data dict missing 'coord' key after transform")

            if isinstance(data_dict["coord"], torch.Tensor):
                self.processed_coord = data_dict["coord"].cpu().numpy()
            else:
                self.processed_coord = data_dict["coord"].copy()

            self.loaded_data = data_dict
            print(f"Cached loaded data for single file mode")
            print(f"Original points: {self.num_total_points}, Processed points: {len(self.processed_coord)}")

    def _create_spatial_blocks_with_kdtree(self):
        """使用KDTree创建空间分块"""
        print(f"Creating spatial blocks with KDTree, {self.num_points_per_block} points per block")

        if self.processed_coord is None:
            if self.current_coord is not None:
                self.processed_coord = self.current_coord
                print("Warning: Using original coordinates for blocking")
            else:
                raise RuntimeError("No processed coordinate data available for blocking")

        num_points = self.processed_coord.shape[0]
        self.current_blocks = []

        if num_points == 0:
            print("No points to process")
            return

        if num_points <= self.num_points_per_block:
            self.current_blocks.append(np.arange(num_points, dtype=np.int32))
            print(f"Only {num_points} points, creating single block")

            if self.save_blocks_to_ply:
                block_coords = self.processed_coord[:]
                self.save_block_to_ply(0, block_coords)
            return

        print(f"Building KDTree for {num_points} points...")
        kdtree = KDTree(self.processed_coord, leaf_size=40)

        effective_points_per_block = int(self.num_points_per_block * (1 - self.overlap_ratio))
        num_blocks = (num_points + effective_points_per_block - 1) // effective_points_per_block
        print(f"Estimated {num_blocks} blocks needed (considering {self.overlap_ratio * 100}% overlap)")

        step = max(1, num_points // num_blocks)
        query_indices = np.arange(0, num_points, step)[:num_blocks]
        print(f"Selected {len(query_indices)} query points")

        for block_id, query_idx in enumerate(query_indices):
            query_point = self.processed_coord[query_idx]
            k = min(self.num_points_per_block, num_points)
            distances, indices = kdtree.query([query_point], k=k)
            block_indices = indices[0]
            block_indices = block_indices[block_indices < num_points]
            self.current_blocks.append(block_indices.astype(np.int32))

            if self.save_blocks_to_ply:
                block_coords = self.processed_coord[block_indices]
                self.save_block_to_ply(block_id, block_coords)

            if (block_id + 1) % 10 == 0 or (block_id + 1) == len(query_indices):
                print(f"Created {block_id + 1}/{len(query_indices)} blocks")

        all_indices = set()
        for block in self.current_blocks:
            all_indices.update(block)

        print(f"Created {len(self.current_blocks)} blocks")
        print(f"Total points in blocks: {sum(len(block) for block in self.current_blocks)}")
        print(f"Unique points covered: {len(all_indices)} / {num_points}")

    def prepare_test_data(self, idx):
        """准备测试数据"""
        if self.test_mode and hasattr(self, 'current_blocks') and len(self.current_blocks) > 0:
            block_idx = idx % len(self.current_blocks)
            block_indices = self.current_blocks[block_idx].copy()

            print(f"Processing block {block_idx}: {len(block_indices)} points")

            if self.single_laz_mode and self.loaded_data is not None:
                full_data = self.loaded_data
            else:
                full_data = self.get_data(idx // len(self.current_blocks))
        else:
            return super().prepare_test_data(idx)

        data_dict = {}
        if isinstance(full_data["coord"], torch.Tensor):
            data_dict["coord"] = full_data["coord"][block_indices].clone()
            data_dict["grid_coord"] = full_data["grid_coord"][block_indices].clone()
            data_dict["inverse"] = full_data["inverse"][block_indices].clone()
            data_dict["feat"] = full_data["feat"][block_indices].clone()
        else:
            data_dict["coord"] = torch.tensor(full_data["coord"][block_indices].copy())

        data_dict["index"] = torch.from_numpy(block_indices.copy())
        data_dict["offset"] = torch.tensor([len(block_indices)], dtype=torch.int32)

        if "segment" in full_data:
            if isinstance(full_data["segment"], torch.Tensor):
                data_dict["segment"] = full_data["segment"][block_indices].clone()
            else:
                data_dict["segment"] = torch.tensor(full_data["segment"][block_indices].copy())
        else:
            data_dict["segment"] = torch.ones(len(block_indices), dtype=torch.int32) * -1

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
            return self.prepare_test_data(idx)
        else:
            return super().__getitem__(idx)

    def __len__(self):
        """数据集长度"""
        if self.test_mode and hasattr(self, 'current_blocks') and len(self.current_blocks) > 0:
            return len(self.current_blocks) * self.loop
        else:
            return super().__len__()

    def save_block_to_ply(self, block_idx, coordinates, filename=None):
        """保存块为PLY格式"""
        if not self.save_blocks_to_ply:
            return

        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()

        if coordinates.ndim == 2 and coordinates.shape[1] == 3:
            pass
        elif coordinates.ndim == 1:
            coordinates = coordinates.reshape(-1, 3)
        else:
            print(f"Warning: Invalid coordinate shape {coordinates.shape} for block {block_idx}")
            return

        if filename is None:
            if self.single_laz_mode and self.laz_file:
                base_name = os.path.splitext(os.path.basename(self.laz_file))[0]
                filename = f"{base_name}_block_{block_idx:04d}.ply"
            else:
                filename = f"block_{block_idx:04d}.ply"

        ply_path = os.path.join(self.ply_save_dir, filename)

        try:
            with open(ply_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(coordinates)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")

                for coord in coordinates:
                    f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")

            print(f"Saved block {block_idx} to PLY file: {ply_path}")
        except Exception as e:
            print(f"Error saving block {block_idx} to PLY: {str(e)}")