"""
Sonata v1m1 Base

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from itertools import chain
from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler
from pointcept.models.sonata.sonata_v1m2_uni_teacher_head import Sonata


import torch
import torch_scatter
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.models.modules import Point


class GenericDensityAugmentor(nn.Module):
    def __init__(
            self,
            num_density_views=3,  # 生成的密度视图数量
            min_ratio=0.3,  # 最小相对密度比例（相对于原始密度）
            max_ratio=3.0,  # 最大相对密度比例
            prob_anisotropic=0.3  # 各向异性采样概率
    ):
        super().__init__()
        self.num_views = num_density_views
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prob_anisotropic = prob_anisotropic

    def forward(self, point):
        """
        生成不同相对密度的点云视图
        不依赖绝对密度计算，通过相对比例缩放原始点数量实现
        """
        batch = offset2batch(point.offset)  # [N]
        num_points_per_batch = torch.bincount(batch)  # 每个批次的原始点数量
        unique_batches = torch.unique(batch)  # 唯一批次索引
        density_views = []

        for _ in range(self.num_views):
            # 为每个批次随机生成密度比例（相对于原始密度）
            ratios = torch.rand(len(unique_batches), device=point.coord.device)
            ratios = ratios * (self.max_ratio - self.min_ratio) + self.min_ratio

            # 各向异性采样（可选）：沿某个轴方向进行非均匀采样
            if self.training and torch.rand(1) < self.prob_anisotropic:
                sampled_indices = self._anisotropic_sample(point, batch, unique_batches, num_points_per_batch, ratios)
            else:
                sampled_indices = self._isotropic_sample(point, batch, unique_batches, num_points_per_batch, ratios)

            # 构建新密度视图
            dense_view = Point({
                "feat": point.feat[sampled_indices],
                "coord": point.coord[sampled_indices],
                "origin_coord": point.origin_coord[sampled_indices],
                "offset": batch2offset(batch[sampled_indices]),
                "grid_size": point.grid_size
            })
            density_views.append(dense_view)

        return density_views

    def _isotropic_sample(self, point, batch, unique_batches, num_points_per_batch, ratios):
        """各向同性采样：均匀降低/增加所有方向的点密度"""
        sampled_points = []

        for i, b in enumerate(unique_batches):
            # 获取当前 batch 的所有点索引
            mask = batch == b
            indices_in_batch = torch.where(mask)[0]

            # 根据比例计算采样数量（至少保留50个点）
            num_sample = max(50, int(ratios[i] * num_points_per_batch[b]))

            # 随机采样
            if num_sample >= len(indices_in_batch):
                # 如果需要的点数大于等于原始点数，直接使用所有点
                selected_indices = indices_in_batch
            else:
                # 随机选择指定数量的点
                rand_indices = torch.randperm(len(indices_in_batch), device=point.coord.device)[:num_sample]
                selected_indices = indices_in_batch[rand_indices]

            sampled_points.append(selected_indices)

        return torch.cat(sampled_points, dim=0)

    def _anisotropic_sample(self, point, batch, unique_batches, num_points_per_batch, ratios):
        """各向异性采样：沿某一轴方向非均匀采样，模拟扫描线密度变化"""
        sampled_points = []
        # 随机选择一个轴（x/y/z）进行非均匀采样
        axis = torch.randint(0, 3, (1,)).item()

        for i, b in enumerate(unique_batches):
            # 获取当前 batch 的所有点索引
            mask = batch == b
            indices_in_batch = torch.where(mask)[0]
            batch_points = point.coord[indices_in_batch]

            # 沿选定轴排序
            sorted_indices_local = torch.argsort(batch_points[:, axis])
            sorted_indices_global = indices_in_batch[sorted_indices_local]

            num_sample = max(50, int(ratios[i] * num_points_per_batch[b]))

            # 非均匀采样：在轴方向上使用不同的采样间隔
            if ratios[i] < 1.0:  # 降采样时，稀疏区域少采，密集区域多采
                # 计算累积分布函数（CDF）实现非均匀采样
                cdf = torch.linspace(0, 1, len(sorted_indices_local), device=point.coord.device)
                cdf = cdf ** (1.0 / ratios[i])  # 调整采样密度曲线
                sample_pos = torch.linspace(0, 1, num_sample, device=point.coord.device)
                indices = torch.searchsorted(cdf, sample_pos).clamp(max=len(sorted_indices_local) - 1)
                selected = sorted_indices_global[indices]
            else:  # 升采样时，在稀疏区域插值补充
                if num_sample <= len(sorted_indices_local):
                    # 如果采样数小于等于原始点数，直接均匀采样
                    indices = torch.linspace(0, len(sorted_indices_local) - 1, num_sample,
                                             device=point.coord.device).long()
                    selected = sorted_indices_global[indices]
                else:
                    # 如果采样数大于原始点数，需要插值或重复采样
                    # 简单实现：重复采样
                    indices = torch.linspace(0, len(sorted_indices_local) - 1, num_sample,
                                             device=point.coord.device).long()
                    indices = indices % len(sorted_indices_local)
                    selected = sorted_indices_global[indices]

            sampled_points.append(selected)

        return torch.cat(sampled_points, dim=0)


@MODELS.register_module("Sonata-v1m2-MD-Generic")
class SonataMultiDensityGeneric(Sonata):
    def __init__(
            self,
            *args,
            num_density_views=3,
            density_min_ratio=0.3,
            density_max_ratio=3.0,
            density_anisotropic_prob=0.3,
            cross_density_weight_start = 0.5,
            cross_density_weight=1.0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # 初始化通用密度增强器（不依赖绝对密度）
        self.density_aug = GenericDensityAugmentor(
            num_density_views=num_density_views,
            min_ratio=density_min_ratio,
            max_ratio=density_max_ratio,
            prob_anisotropic=density_anisotropic_prob
        )
        self.cross_density_loss = CrossDensityLoss()
        self.cross_density_weight = cross_density_weight
        self.cross_density_weight_start = cross_density_weight_start

    def before_train(self):
        super().before_train()
        # 密度损失权重调度器
        total_steps = self.trainer.cfg.scheduler.total_steps
        self.density_weight_scheduler = CosineScheduler(
            start_value=self.cross_density_weight_start,
            base_value=self.cross_density_weight,
            final_value=self.cross_density_weight,
            total_iters=total_steps
        )

    def before_step(self):
        super().before_step()
        self.current_density_weight = self.density_weight_scheduler.step()

    def forward(self, data_dict, return_point=False):
        if return_point:
            return super().forward(data_dict, return_point)

        # 1. 生成多密度视图（基于原始点云的相对密度）
        global_point = Point(
            feat=data_dict["global_feat"],
            coord=data_dict["global_coord"],
            origin_coord=data_dict["global_origin_coord"],
            offset=data_dict["global_offset"],
            grid_size=data_dict["grid_size"][0],
        )
        density_views = self.density_aug(global_point)

        # 2. 原有损失计算
        base_result = super().forward(data_dict)

        # 3. 跨密度一致性损失计算
        # with torch.no_grad():
        #     teacher_feats = [
        #         self.teacher.mask_head(self.up_cast(self.teacher.backbone(view)))
        #         for view in density_views
        #     ]
        #
        # student_feats = [
        #     self.student.mask_head(self.up_cast(self.student.backbone(view)).feat)
        #     for view in density_views
        # ]
        with torch.no_grad():
            teacher_data_dict = [
                # self.teacher.mask_head(self.up_cast(self.teacher.backbone(view)).feat)
                self.up_cast(self.teacher.backbone(view))
                for view in density_views
            ]
        student_data_dict = [
            # self.student.mask_head(self.up_cast(self.student.backbone(view)).feat)
            self.up_cast(self.student.backbone(view))
            for view in density_views
        ]
        student_feats = [
            student_data_dict[i]["feat"]
            for i in range(len(density_views))
        ]
        teacher_feats = [
            teacher_data_dict[i]["feat"]
            for i in range(len(density_views))
        ]
        # 提取坐标信息用于匹配（可选）
        student_coord_list = [
            student_data_dict[i]["coord"]
            for i in range(len(density_views))
        ]
        teacher_coord_list = [
            teacher_data_dict[i]["coord"]
            for i in range(len(density_views))
        ]
        # coord_list = [view.coord for view in density_views]

        # 计算不同密度视图间的特征一致性损失
        cross_loss = self.cross_density_loss(student_feats, student_coord_list)
        with torch.no_grad():
            teacher_cross_loss = self.cross_density_loss(teacher_feats, teacher_coord_list)
        cross_loss = (cross_loss + teacher_cross_loss) * 0.5

        # 4. 合并损失
        base_result["cross_density_loss"] = cross_loss * self.current_density_weight
        base_result["loss"] += base_result["cross_density_loss"]

        return base_result


import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.utils import offset2batch, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
import pointops  # 导入pointops库


class CrossDensityLoss(nn.Module):
    """
    跨密度视图特征一致性损失
    通过动态匹配不同密度视图间的点，并计算特征相似性损失
    基于pointops库实现高效的KNN查询
    """

    def __init__(
            self,
            temp=0.1,
            match_max_k=8,  # 每个点匹配的最大邻居数
            sinkhorn_iter=3,
    ):
        super().__init__()
        self.temp = temp
        self.match_max_k = match_max_k  # 每个点匹配的最大邻居数
        self.sinkhorn_iter = sinkhorn_iter  # Sinkhorn-Knopp迭代次数

    def forward(self, feat_list, coord_list, offset_list=None):
        """
        Args:
            feat_list: 不同密度视图的特征列表，每个元素形状为[N_i, C]
            coord_list: 不同密度视图的坐标列表，每个元素形状为[N_i, 3]
            offset_list: 不同密度视图的offset列表，每个元素形状为[B+1]，用于批次区分
        Returns:
            跨密度视图一致性损失
        """
        total_loss = 0.0
        num_views = len(feat_list)

        # 计算所有视图对之间的一致性损失
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # 获取当前视图对的offset（如果提供）
                offset_i = offset_list[i] if offset_list is not None else None
                offset_j = offset_list[j] if offset_list is not None else None

                loss_ij = self._view_pair_loss(
                    feat_i=feat_list[i],
                    coord_i=coord_list[i],
                    feat_j=feat_list[j],
                    coord_j=coord_list[j],
                    offset_i=offset_i,
                    offset_j=offset_j
                )
                total_loss += loss_ij

        # 平均所有视图对的损失
        return total_loss / (num_views * (num_views - 1) / 2)

    def _view_pair_loss(self, feat_i, coord_i, feat_j, coord_j, offset_i=None, offset_j=None):
        """计算两个视图之间的跨密度损失"""
        # 特征归一化
        feat_i = F.normalize(feat_i, dim=1)
        feat_j = F.normalize(feat_j, dim=1)

        # 为没有提供offset的情况自动生成（假设单批次）
        if offset_i is None:
            offset_i = torch.tensor([0, coord_i.size(0)], device=coord_i.device, dtype=torch.int32)
        if offset_j is None:
            offset_j = torch.tensor([0, coord_j.size(0)], device=coord_j.device, dtype=torch.int32)

        # 动态匹配两个视图中的点 - 使用pointops的knnquery
        idx_j, dists_j = pointops.knn_query(
            self.match_max_k,  # 邻居数量
            coord_j.contiguous().float(),  # 目标点云
            offset_j.contiguous().int(),  # 查询点云（从i查询j中的点）
            coord_i.contiguous().float(),  # 目标点云的offset
            offset_i.contiguous().int()   # 查询点云的offset
        )  # idx_j: [N_i, K], dists_j: [N_i, K]

        # 获取匹配点的特征 (使用pointops的grouping函数)
        feat_j_matched = pointops.grouping(idx_j.contiguous(), feat_j.contiguous(), coord_j.contiguous())  # [N_i, K, C]

        # 计算特征相似性
        sim_matrix = torch.einsum(
            "nc,nkc->nk",
            feat_i,
            feat_j_matched
        )  # [N_i, K]
        sim_matrix = sim_matrix / self.temp

        # 使用Sinkhorn-Knopp算法计算最优匹配
        q_i = self.sinkhorn_knopp(sim_matrix, temp=1.0)  # [N_i, K]

        # 计算InfoNCE损失
        loss_i = -torch.log(torch.sum(q_i * F.softmax(sim_matrix, dim=1), dim=1) + 1e-12)
        loss_i = loss_i.mean()

        # 对称计算损失（j->i）
        idx_i, dists_i = pointops.knn_query(
            self.match_max_k,  # 邻居数量
            coord_i.contiguous(),  # 目标点云
            offset_i.contiguous(),  # 查询点云（从j查询i中的点）
            coord_j.contiguous(),  # 目标点云的offset
            offset_j.contiguous()   # 查询点云的offset
        )  # idx_i: [N_j, K], dists_i: [N_j, K]

        # 获取匹配点的特征

        feat_i_matched = pointops.grouping(idx_i.contiguous(), feat_i.contiguous(), coord_i.contiguous())
        # 计算特征相似性
        sim_matrix_j = torch.einsum(
            "nc,nkc->nk",
            feat_j,
            feat_i_matched
        )  # [N_j, K]
        sim_matrix_j = sim_matrix_j / self.temp

        q_j = self.sinkhorn_knopp(sim_matrix_j, temp=1.0)
        loss_j = -torch.log(torch.sum(q_j * F.softmax(sim_matrix_j, dim=1), dim=1) + 1e-12)
        loss_j = loss_j.mean()

        return (loss_i + loss_j) * 0.5

    @staticmethod
    def sinkhorn_knopp(feat, temp=1.0, num_iter=3):
        """Sinkhorn-Knopp算法用于计算最优传输矩阵"""
        feat = feat.float()
        q = torch.exp(feat / temp).t()  # [K, N]
        n = sum(all_gather(q.shape[1]))  # 全局样本数
        k = q.shape[0]  # 原型数

        # 归一化
        sum_q = q.sum()
        if get_world_size() > 1:
            torch.distributed.all_reduce(sum_q)
        q = q / sum_q

        for _ in range(num_iter):
            # 行归一化
            sum_r = torch.sum(q, dim=1, keepdim=True)
            if get_world_size() > 1:
                torch.distributed.all_reduce(sum_r)
            q = q / sum_r

            # 列归一化
            sum_c = torch.sum(q, dim=0, keepdim=True)
            q = q / sum_c

        return q.t()  # [N, K]