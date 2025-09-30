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
# from pointcept.models.sonata.sonata_v1m2_uni_teacher_head import Sonata
from pointcept.models.sonata.sonata_v1m1_base import Sonata


import torch
import torch_scatter
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.models.modules import Point


class GenericDensityAugmentor(nn.Module):
    def __init__(
            self,
            num_density_views=2,  # 减少密度视图数量，从3减到2
            min_ratio=0.3,
            max_ratio=2.0,
            prob_anisotropic=0.3
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

            # 各向异性采样（可选）
            if self.training and torch.rand(1) < self.prob_anisotropic:
                sampled_indices = self._anisotropic_sample(point, batch, unique_batches, num_points_per_batch, ratios)
            else:
                sampled_indices = self._isotropic_sample(point, batch, unique_batches, num_points_per_batch, ratios)

            # 构建新密度视图 - 只保留必要的字段
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
                selected_indices = indices_in_batch
            else:
                # 使用更高效的随机采样方法
                rand_indices = torch.randperm(len(indices_in_batch), device=point.coord.device, dtype=torch.int64)[:num_sample]
                selected_indices = indices_in_batch[rand_indices]

            sampled_points.append(selected_indices)

        return torch.cat(sampled_points, dim=0)

    def _anisotropic_sample(self, point, batch, unique_batches, num_points_per_batch, ratios):
        """各向异性采样：沿某一轴方向非均匀采样"""
        sampled_points = []
        axis = torch.randint(0, 3, (1,)).item()

        for i, b in enumerate(unique_batches):
            mask = batch == b
            indices_in_batch = torch.where(mask)[0]
            batch_points = point.coord[indices_in_batch]

            # 沿选定轴排序
            sorted_indices_local = torch.argsort(batch_points[:, axis])
            sorted_indices_global = indices_in_batch[sorted_indices_local]

            num_sample = max(50, int(ratios[i] * num_points_per_batch[b]))

            # 简化非均匀采样逻辑
            if ratios[i] < 1.0:
                # 降采样时使用均匀间隔采样
                step = max(1, len(sorted_indices_local) // num_sample)
                selected = sorted_indices_global[::step][:num_sample]
            else:
                # 升采样时使用重复采样
                indices = torch.linspace(0, len(sorted_indices_local) - 1, num_sample,
                                         device=point.coord.device, dtype=torch.int64)
                indices = indices % len(sorted_indices_local)
                selected = sorted_indices_global[indices]

            sampled_points.append(selected)

        return torch.cat(sampled_points, dim=0)


@MODELS.register_module("Sonata-v1m2-MD-Generic")
class SonataMultiDensityPointLevel(Sonata):
    def __init__(
            self,
            *args,
            density_min_ratio=0.8,  # 初始扰动范围小
            density_max_ratio=1.2,  # 初始扰动范围小
            density_consistency_weight_start=0.01,  # 低初始权重
            density_consistency_weight=0.2,  # 适度目标权重
            density_radius=0.1,  # 局部区域半径
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        # 初始化密度扰动增强器
        self.density_aug = GenericDensityAugmentor(
            num_density_views=2,
            min_ratio=density_min_ratio,
            max_ratio=density_max_ratio
        )

        # 初始化点级别密度不变损失
        self.density_consistency_loss = PointLevelDensityConsistencyLoss(
            temp=0.1,
            radius=density_radius
        )

        # 设置损失权重参数
        self.density_consistency_weight = density_consistency_weight
        self.density_consistency_weight_start = density_consistency_weight_start

    def before_train(self):
        super().before_train()
        total_steps = self.trainer.cfg.scheduler.total_steps

        # 密度一致性损失权重调度器（缓慢增长）
        self.density_weight_scheduler = CosineScheduler(
            start_value=self.density_consistency_weight_start,
            base_value=self.density_consistency_weight,
            final_value=self.density_consistency_weight,
            total_iters=total_steps * 0.5
        )

        # 密度扰动范围调度器（渐进式）
        self.density_range_scheduler = CosineScheduler(
            start_value=0.0,  # 无扰动
            base_value=1.0,  # 完全达到目标扰动范围
            final_value=1.0,
            total_iters=int(total_steps * 0.3)
        )

    def before_step(self):
        super().before_step()
        # 更新当前密度损失权重
        self.current_density_weight = self.density_weight_scheduler.step()

        # 更新密度扰动范围
        # if hasattr(self, 'density_range_scheduler'):
        #     current_ratio = self.trainer.iter / self.trainer.max_iters
        #     current_min_ratio = 1.0 - (1.0 - self.density_aug.min_ratio) * min(current_ratio / 0.3, 1.0)
        #     current_max_ratio = 1.0 + (self.density_aug.max_ratio - 1.0) * min(current_ratio / 0.3, 1.0)
        #     self.density_aug.min_ratio = current_min_ratio
        #     self.density_aug.max_ratio = current_max_ratio

    def forward(self, data_dict, return_point=False):
        if return_point:
            return super().forward(data_dict, return_point)

        # 1. 生成多密度视图
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

        # 3. 密度不变损失计算
        if self.training and len(density_views) > 1:
            # 获取教师模型特征（作为稳定目标）
            with torch.no_grad():
                teacher_feats = []
                teacher_coords = []
                for view in density_views:
                    teacher_out = self.up_cast(self.teacher.backbone(view))
                    teacher_feats.append(teacher_out["feat"])
                    teacher_coords.append(teacher_out["coord"])

            # 获取学生模型特征
            student_feats = []
            student_coords = []
            for view in density_views:
                student_out = self.up_cast(self.student.backbone(view))
                student_feats.append(student_out["feat"])
                student_coords.append(student_out["coord"])

            # 计算点级别密度不变损失
            density_loss = self.density_consistency_loss(
                student_feats,
                teacher_feats,
                student_coords,
                [view.offset for view in density_views]
            )

            # 合并到总损失
            base_result["density_consistency_loss"] = density_loss
            base_result["loss"] += density_loss * self.current_density_weight

        return base_result
class PointLevelDensityConsistencyLoss(nn.Module):
    def __init__(self, temp=0.1, radius=0.1, density_norm_power=0.5):
        """
        点级别密度不变语义一致性损失

        Args:
            temp: 对比学习温度参数
            radius: 局部区域半径 (根据数据集调整，ScanNet建议0.1)
            density_norm_power: 密度归一化指数 (0.5表示平方根归一化)
        """
        super().__init__()
        self.temp = temp
        self.radius = radius
        self.density_norm_power = density_norm_power

        # 投影头，将特征映射到对比学习空间
        self.projection_head = nn.Sequential(
            nn.Linear(1088, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )

    def forward(self, student_feats, teacher_feats, coords, offsets=None):
        """
        计算点级别密度不变语义一致性损失

        Args:
            student_feats: 学生模型在不同密度视图的特征列表 [num_views, N_i, C]
            teacher_feats: 教师模型在不同密度视图的特征列表 [num_views, N_i, C]
            coords: 原始坐标列表 [num_views, N_i, 3]
            offsets: 批次偏移列表 [num_views, B+1]

        Returns:
            密度一致性损失值
        """
        num_views = len(student_feats)
        total_loss = 0.0
        valid_pairs = 0

        # 遍历所有视图对
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # 仅处理有效的视图对
                if len(student_feats[i]) == 0 or len(student_feats[j]) == 0:
                    continue

                # 投影特征并归一化
                student_proj_i = F.normalize(self.projection_head(student_feats[i]), dim=-1)
                student_proj_j = F.normalize(self.projection_head(student_feats[j]), dim=-1)
                teacher_proj_i = F.normalize(self.projection_head(teacher_feats[i]), dim=-1)
                teacher_proj_j = F.normalize(self.projection_head(teacher_feats[j]), dim=-1)

                # 计算视图i→j的密度不变损失
                loss_ij = self._density_invariant_loss(
                    student_proj_i, teacher_proj_j,
                    coords[i], coords[j],
                    offsets[i] if offsets else None,
                    offsets[j] if offsets else None
                )

                # 计算视图j→i的密度不变损失
                loss_ji = self._density_invariant_loss(
                    student_proj_j, teacher_proj_i,
                    coords[j], coords[i],
                    offsets[j] if offsets else None,
                    offsets[i] if offsets else None
                )

                total_loss += (loss_ij + loss_ji) * 0.5
                valid_pairs += 1

        return total_loss / max(1, valid_pairs)

    def _density_invariant_loss(self, student_proj, teacher_proj,
                                coord_s, coord_t, offset_s=None, offset_t=None):
        """计算点级别密度不变损失"""
        # 1. 处理offset缺失情况
        if offset_s is None:
            offset_s = torch.tensor([0, coord_s.size(0)], device=coord_s.device, dtype=torch.int32)
        if offset_t is None:
            offset_t = torch.tensor([0, coord_t.size(0)], device=coord_t.device, dtype=torch.int32)

        # 2. 密度归一化局部特征聚合 (关键步骤)
        density_invariant_s = self._density_normalized_aggregation(
            student_proj, coord_s, offset_s
        )
        density_invariant_t = self._density_normalized_aggregation(
            teacher_proj, coord_t, offset_t
        )

        # 3. 计算相似度矩阵
        sim_matrix = torch.mm(density_invariant_s, density_invariant_t.t()) / self.temp

        # 4. 生成软匹配标签 (基于几何邻近性)
        with torch.no_grad():
            # 计算点s到点t的最近邻距离
            dist_matrix = self._compute_distance_matrix(coord_s, coord_t, offset_s, offset_t)

            # 创建软标签：考虑多个潜在匹配点
            _, topk_indices = torch.topk(-dist_matrix, k=min(3, dist_matrix.size(1)), dim=1)
            labels = torch.zeros_like(sim_matrix)

            # 为每个查询点分配权重
            for i in range(dist_matrix.size(0)):
                weights = F.softmax(-dist_matrix[i, topk_indices[i]], dim=0)
                labels[i, topk_indices[i]] = weights

        # 5. 计算对比损失
        log_prob = F.log_softmax(sim_matrix, dim=1)
        loss = -torch.sum(labels * log_prob, dim=1).mean()

        return loss

    def _density_normalized_aggregation(self, feat, coord, offset):
        """密度归一化的局部特征聚合 (关键创新)"""
        # 1. 为每个点查询局部球形邻域
        idx, _ = pointops.knn_query(4, coord, offset, coord, offset)

        # 2. 获取邻域特征
        grouped_feat = pointops.grouping(idx.contiguous(), feat.contiguous(), coord.contiguous())

        # 3. 密度归一化：根据邻域点数调整聚合权重
        #valid_mask = (idx >= 0)
        valid_mask = idx
        num_points = valid_mask.sum(dim=1, keepdim=True).float()  # [N, 1]

        # 关键：使用幂函数进行密度归一化 (0.5表示平方根归一化)
        normalized_weights = valid_mask.float() / (num_points ** self.density_norm_power + 1e-6)

        # 4. 密度加权聚合
        density_invariant_feat = torch.sum(grouped_feat * normalized_weights.unsqueeze(-1), dim=1)

        return density_invariant_feat

    def _compute_distance_matrix(self, coord_s, coord_t, offset_s=None, offset_t=None):
        """计算距离矩阵，考虑批次信息并修复维度不匹配问题"""
        # 确保offset有效
        if offset_s is None or len(offset_s) == 0:
            offset_s = torch.tensor([0, coord_s.size(0)], device=coord_s.device, dtype=torch.int32)
        if offset_t is None or len(offset_t) == 0:
            offset_t = torch.tensor([0, coord_t.size(0)], device=coord_t.device, dtype=torch.int32)

        # 创建正确的批次索引（避免使用可能有问题的offset2batch）
        batch_s = torch.zeros(coord_s.size(0), dtype=torch.long, device=coord_s.device)
        batch_t = torch.zeros(coord_t.size(0), dtype=torch.long, device=coord_t.device)

        # 手动构建批次索引
        for i in range(1, len(offset_s)):
            start, end = offset_s[i - 1], offset_s[i]
            if end > start:  # 确保有效范围
                batch_s[start:end] = i - 1

        for i in range(1, len(offset_t)):
            start, end = offset_t[i - 1], offset_t[i]
            if end > start:
                batch_t[start:end] = i - 1

        # 创建距离矩阵
        dist_matrix = torch.zeros(coord_s.size(0), coord_t.size(0),
                                  device=coord_s.device, dtype=coord_s.dtype)

        # 处理每个批次
        max_batch = max(batch_s.max().item(), batch_t.max().item()) + 1
        for b in range(max_batch):
            mask_s = (batch_s == b)
            mask_t = (batch_t == b)

            # 检查有效点
            if mask_s.sum() == 0 or mask_t.sum() == 0:
                continue

            # 安全计算距离
            try:
                batch_dist = torch.cdist(coord_s[mask_s], coord_t[mask_t])
                dist_matrix[mask_s][:, mask_t] = batch_dist
            except Exception as e:
                print(f"Error computing distance for batch {b}: {e}")
                continue

        return dist_matrix



class CrossDensityLoss(nn.Module):
    """
    简化的跨密度视图特征一致性损失
    减少计算复杂度和内存占用
    """

    def __init__(
            self,
            temp=0.1,
            match_max_k=4,  # 减少KNN邻居数量，从8减到4
            use_sinkhorn=False,  # 默认不使用Sinkhorn-Knopp算法
    ):
        super().__init__()
        self.temp = temp
        self.match_max_k = match_max_k
        self.use_sinkhorn = use_sinkhorn

    def forward(self, feat_list, coord_list, offset_list=None):
        """
        Args:
            feat_list: 不同密度视图的特征列表
            coord_list: 不同密度视图的坐标列表
            offset_list: 不同密度视图的offset列表
        Returns:
            跨密度视图一致性损失
        """
        total_loss = 0.0
        num_views = len(feat_list)

        # 只计算相邻视图对之间的损失，减少计算量
        for i in range(num_views - 1):
            j = i + 1
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
        return total_loss / max(1, num_views - 1)

    def _view_pair_loss(self, feat_i, coord_i, feat_j, coord_j, offset_i=None, offset_j=None):
        """计算两个视图之间的跨密度损失"""
        # 特征归一化
        feat_i = F.normalize(feat_i, dim=1)
        feat_j = F.normalize(feat_j, dim=1)

        # 为没有提供offset的情况自动生成
        if offset_i is None:
            offset_i = torch.tensor([0, coord_i.size(0)], device=coord_i.device, dtype=torch.int32)
        if offset_j is None:
            offset_j = torch.tensor([0, coord_j.size(0)], device=coord_j.device, dtype=torch.int32)

        # 使用pointops进行KNN查询
        idx_j, _ = pointops.knn_query(
            self.match_max_k,
            coord_j.contiguous().float(),
            offset_j.contiguous().int(),
            coord_i.contiguous().float(),
            offset_i.contiguous().int()
        )

        # 获取匹配点的特征
        feat_j_matched = pointops.grouping(idx_j.contiguous(), feat_j.contiguous(), coord_j.contiguous())

        # 计算特征相似性
        sim_matrix = torch.einsum("nc,nkc->nk", feat_i, feat_j_matched)
        sim_matrix = sim_matrix / self.temp

        # 简化匹配策略：使用softmax直接匹配
        if self.use_sinkhorn:
            q_i = self.sinkhorn_knopp(sim_matrix, temp=1.0)
        else:
            q_i = F.softmax(sim_matrix, dim=1)

        # 计算InfoNCE损失
        loss_i = -torch.log(torch.sum(q_i * F.softmax(sim_matrix, dim=1), dim=1) + 1e-12)
        loss_i = loss_i.mean()

        # 对称计算损失（j->i）
        idx_i, _ = pointops.knn_query(
            self.match_max_k,
            coord_i.contiguous().float(),
            offset_i.contiguous().int(),
            coord_j.contiguous().float(),
            offset_j.contiguous().int()
        )

        feat_i_matched = pointops.grouping(idx_i.contiguous(), feat_i.contiguous(), coord_i.contiguous())
        sim_matrix_j = torch.einsum("nc,nkc->nk", feat_j, feat_i_matched)
        sim_matrix_j = sim_matrix_j / self.temp

        if self.use_sinkhorn:
            q_j = self.sinkhorn_knopp(sim_matrix_j, temp=1.0)
        else:
            q_j = F.softmax(sim_matrix_j, dim=1)

        loss_j = -torch.log(torch.sum(q_j * F.softmax(sim_matrix_j, dim=1), dim=1) + 1e-12)
        loss_j = loss_j.mean()

        return (loss_i + loss_j) * 0.5

    @staticmethod
    def sinkhorn_knopp(feat, temp=1.0, num_iter=2):  # 减少迭代次数
        """简化的Sinkhorn-Knopp算法"""
        feat = feat.float()
        q = torch.exp(feat / temp).t()  # [K, N]

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

        return q.t()