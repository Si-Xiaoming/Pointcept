import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping

from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")
from .transform import Compose

@TRANSFORMS.register_module()
class DensityPerturbationViewGenerator(object):
    def __init__(
            self,
            global_view_num=2,
            global_view_size=(10.0, 30.0),
            local_view_num=4,
            local_view_size=(2.0, 10.0),
            global_shared_transform=None,
            global_transform=None,
            local_transform=None,
            max_size=65536,
            center_height_scale=(0.2, 0.8),
            shared_global_view=False,
            view_keys=("coord", "origin_coord", "intensity"),
            shape_type="cube",
            # 密度扰动参数 - 简化为固定范围
            density_perturbation=True,
            density_perturbation_prob=0.8,
            density_variation=(0.1, 10.0),  # 直接表示密度变化范围
            density_estimation_method="knn",
            density_k=20
    ):
        # 基本参数初始化
        self.global_view_num = global_view_num
        self.global_view_size = global_view_size
        self.local_view_num = local_view_num
        self.local_view_size = local_view_size
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.max_size = max_size
        self.center_height_scale = center_height_scale
        self.shared_global_view = shared_global_view
        self.view_keys = view_keys
        self.shape_type = shape_type
        assert "coord" in view_keys

        # 密度扰动参数
        self.density_perturbation = density_perturbation
        self.density_perturbation_prob = density_perturbation_prob
        self.density_variation = density_variation
        self.density_estimation_method = density_estimation_method
        self.density_k = density_k

    def estimate_density(self, coord, center):
        """密度估计 - 简化实现"""
        try:
            from sklearn.neighbors import KDTree
            kdtree = KDTree(coord[:, :2])
            distances, _ = kdtree.query([center[:2]], k=min(self.density_k, len(coord)))
            k_dist = distances[0][-1]
            # 密度 = k / (π * r²)
            return self.density_k / (np.pi * k_dist ** 2 + 1e-8)
        except:
            # 简单回退方法
            return len(coord) / (1.0 + 1e-8)  # 避免除零

    def interpolate_point_cloud(self, point, indices, num_to_add):
        """基于插值的上采样"""
        if num_to_add <= 0 or len(indices) == 0:
            return indices

        coord = point["coord"]
        try:
            from sklearn.neighbors import KDTree
            kdtree = KDTree(coord[:, :2])

            new_points = []
            for _ in range(num_to_add):
                # 随机选择一个基础点
                base_idx = np.random.choice(indices)
                base_point = coord[base_idx]

                # 找到最近邻
                _, neighbor_idx = kdtree.query([base_point[:2]], k=min(10, len(coord)))
                neighbor_points = coord[neighbor_idx[0]]

                # 随机选择两个邻居进行线性插值
                idx1, idx2 = np.random.choice(len(neighbor_points), 2, replace=False)
                weight = np.random.rand()
                new_point = weight * neighbor_points[idx1] + (1 - weight) * neighbor_points[idx2]
                # 保持Z坐标在合理范围内
                new_point[2] = base_point[2] + np.random.normal(0, 0.1)
                new_points.append(new_point)
        except:
            # 简单回退方法
            base_indices = np.random.choice(indices, num_to_add, replace=True)
            new_points = coord[base_indices]

        new_points = np.array(new_points)

        # 临时扩展坐标
        new_coord = np.vstack([coord, new_points])
        point["coord"] = new_coord

        # 处理其他特征
        for key in self.view_keys:
            if key != "coord" and key in point and len(point[key]) == len(coord):
                # 复制基础点的特征
                base_features = point[key][np.random.choice(indices, num_to_add, replace=True)]
                point[key] = np.vstack([point[key], base_features])

        # 返回扩展后的索引
        return np.concatenate([indices, np.arange(len(coord), len(new_coord))])

    def perturb_density(self, point, indices):
        """实现密度扰动 - 作为固定数据预处理"""
        if not self.density_perturbation or np.random.rand() > self.density_perturbation_prob or len(indices) == 0:
            return indices

        # 获取当前区域的密度
        current_center = np.mean(point["coord"][indices], axis=0)
        current_density = self.estimate_density(point["coord"][indices], current_center)

        # 随机选择密度变化因子 - 直接表示目标密度是当前密度的多少倍
        density_factor = np.random.uniform(*self.density_variation)

        # 处理下采样 (density_factor < 1.0)
        if density_factor < 1.0:
            num_to_sample = int(len(indices) * density_factor)
            num_to_sample = max(10, min(num_to_sample, self.max_size))
            return np.random.choice(indices, num_to_sample, replace=False)

        # 处理上采样 (density_factor > 1.0)
        elif density_factor > 1.0:
            num_to_add = min(int(len(indices) * (density_factor - 1.0)),
                             self.max_size - len(indices))
            if num_to_add > 0:
                return self.interpolate_point_cloud(point, indices, num_to_add)

        # 如果不需要改变密度，但超过max_size
        if len(indices) > self.max_size:
            return np.random.choice(indices, self.max_size, replace=False)

        return indices

    def get_view_by_size(self, point, center, size):
        """获取基于物理尺寸的视图，包含密度扰动"""
        coord = point["coord"]

        # 计算距离中心点在指定物理尺寸范围内的点
        if self.shape_type == "cube":
            x_mask = np.abs(coord[:, 0] - center[0]) < size / 2
            y_mask = np.abs(coord[:, 1] - center[1]) < size / 2
            mask = np.logical_and(x_mask, y_mask)
        else:  # sphere
            distances = np.sqrt(np.sum((coord[:, :2] - center[:2]) ** 2, axis=1))
            mask = distances < size / 2

        # 应用高度过滤
        if len(coord) > 0 and coord.shape[1] >= 3:
            z_min = np.min(coord[:, 2])
            z_max = np.max(coord[:, 2])
            z_range = z_max - z_min
            z_min_ = z_min + z_range * self.center_height_scale[0]
            z_max_ = z_min + z_range * self.center_height_scale[1]
            z_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)
            mask = np.logical_and(mask, z_mask)

        indices = np.where(mask)[0]

        # 如果点太少，返回空视图
        if len(indices) == 0:
            return None

        # 应用密度扰动 - 作为固定数据预处理
        indices = self.perturb_density(point, indices)

        # 创建视图
        view = dict(index=indices)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][indices]

        if "index_valid_keys" in point.keys():
            view["index_valid_keys"] = point["index_valid_keys"]

        return view

    def __call__(self, data_dict):
        # 原始视图生成逻辑保持不变
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))

        # 创建高度掩码
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)

        # 选择一个合理的中心点（避免边缘）
        valid_indices = np.where(center_mask)[0]
        if len(valid_indices) == 0:
            return data_dict

        # 确保中心点不在边缘
        max_global_size = max(self.global_view_size)
        edge_buffer = max_global_size / 2
        x_min, y_min = np.min(coord[:, :2], axis=0)
        x_max, y_max = np.max(coord[:, :2], axis=0)

        # 过滤掉靠近边缘的点
        edge_mask = np.logical_and(
            np.logical_and(coord[:, 0] >= x_min + edge_buffer, coord[:, 0] <= x_max - edge_buffer),
            np.logical_and(coord[:, 1] >= y_min + edge_buffer, coord[:, 1] <= y_max - edge_buffer)
        )
        center_mask = np.logical_and(center_mask, edge_mask)

        valid_indices = np.where(center_mask)[0]
        if len(valid_indices) == 0:
            # 如果没有足够远离边缘的点，放宽条件
            valid_indices = np.where(np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_))[0]

        if len(valid_indices) == 0:
            return data_dict

        # 获取主要全局视图
        major_center = coord[np.random.choice(valid_indices)]
        global_size = np.random.uniform(*self.global_view_size)
        major_view = self.get_view_by_size(point, major_center, global_size)

        if major_view is None:
            return data_dict

        major_coord = major_view["coord"]

        # 获取其他全局视图
        global_views = []
        for _ in range(self.global_view_num - 1):
            if major_coord.shape[0] == 0:
                break
            center_idx = np.random.randint(major_coord.shape[0])
            center = major_coord[center_idx]
            size = np.random.uniform(*self.global_view_size)
            view = self.get_view_by_size(point, center, size)
            if view is not None:
                global_views.append(view)

        if len(global_views) < self.global_view_num - 1:
            # 如果无法生成足够的全局视图，用major_view填充
            while len(global_views) < self.global_view_num - 1:
                global_views.append({key: value.copy() for key, value in major_view.items()})

        global_views = [major_view] + global_views

        # 获取局部视图
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        for i in range(self.local_view_num):
            if sum(~cover_mask) == 0:
                # 重置覆盖掩码
                cover_mask[:] = False

            if major_coord.shape[0] == 0:
                break

            # 优先选择未覆盖区域的点
            available_indices = np.where(~cover_mask)[0]
            if len(available_indices) == 0:
                available_indices = np.arange(major_coord.shape[0])

            center_idx = np.random.choice(available_indices)
            center = major_coord[center_idx]
            size = np.random.uniform(*self.local_view_size)
            local_view = self.get_view_by_size(point, center, size)

            if local_view is not None:
                local_views.append(local_view)
                # 更新覆盖掩码
                local_indices = local_view["index"]
                major_indices = major_view["index"]
                mask = np.isin(major_indices, local_indices)
                cover_mask[mask] = True

        # 应用变换和拼接
        view_dict = {}
        for global_view in global_views:
            if global_view is None:
                continue
            global_view.pop("index")
            global_view = self.global_transform(global_view)
            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]

        if "global_coord" in view_dict and len(view_dict["global_coord"]) > 0:
            view_dict["global_offset"] = np.cumsum(
                [data.shape[0] for data in view_dict["global_coord"]]
            )

        for local_view in local_views:
            if local_view is None:
                continue
            local_view.pop("index")
            local_view = self.local_transform(local_view)
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]

        if "local_coord" in view_dict and len(view_dict["local_coord"]) > 0:
            view_dict["local_offset"] = np.cumsum(
                [data.shape[0] for data in view_dict["local_coord"]]
            )

        for key in view_dict.keys():
            if "offset" not in key and len(view_dict[key]) > 0:
                view_dict[key] = np.concatenate(view_dict[key], axis=0)

        data_dict.update(view_dict)
        return data_dict