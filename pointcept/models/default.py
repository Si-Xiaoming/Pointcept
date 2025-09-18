import torch
import torch.nn as nn
import torch_scatter
import torch_cluster

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module()
class DefaultSegmentorV3(nn.Module):
    # fine tuning
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        
        
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"], point.feat, point.batch)
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"], point.feat, point.batch)
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


class PointCFPFusionModule(nn.Module):
    """点云版CFP特征融合模块"""

    def __init__(self, in_channels, reduction_ratio=4):
        super(PointCFPFusionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        mid_channels = max(1, in_channels // reduction_ratio)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Linear(in_channels + 3, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

        # 特征融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_res_feat, high_res_coord, low_res_feat, low_res_coord,
                high_res_offset):
        """融合高低分辨率特征"""
        # 计算低分辨率特征到高分辨率特征的最近邻映射
        inverse_indices = self._compute_nearest_neighbor_mapping(
            high_res_coord, low_res_coord
        )

        # 上采样低分辨率特征
        low_res_up = low_res_feat[inverse_indices]

        # 通道注意力
        batch_idx = offset2batch(high_res_offset)
        global_feat = torch_scatter.scatter_mean(low_res_up, batch_idx, dim=0)
        channel_weights = self.channel_attention(global_feat)[batch_idx]
        channel_refined = high_res_feat * channel_weights

        # 空间注意力
        feat_with_coord = torch.cat([channel_refined, high_res_coord], dim=1)
        spatial_weights = self.spatial_attention(feat_with_coord)
        spatial_refined = channel_refined * spatial_weights

        # 特征融合
        fused_feat = torch.cat([high_res_feat, spatial_refined], dim=1)
        return self.fusion_mlp(fused_feat)

    @staticmethod
    def _compute_nearest_neighbor_mapping(high_res_coord, low_res_coord):
        """计算高分辨率点到低分辨率点的最近邻映射"""
        if high_res_coord.device != low_res_coord.device:
            low_res_coord = low_res_coord.to(high_res_coord.device)

        # 计算距离矩阵
        dist_matrix = torch.cdist(high_res_coord.unsqueeze(0), low_res_coord.unsqueeze(0)).squeeze(0)

        # 找到每个高分辨率点对应的最近低分辨率点
        nearest_indices = torch.argmin(dist_matrix, dim=1)

        return nearest_indices


class MultiResolutionPointGenerator(nn.Module):
    """多分辨率点云生成器（修复空列表问题）"""

    def __init__(self, scales=[1.0, 0.5, 0.25], sampling_method='fps'):
        super().__init__()
        self.scales = scales
        self.sampling_method = sampling_method
        # 添加最小采样点数限制，避免采样后点数量为0
        self.min_samples = 1

    def forward(self, input_dict):
        """生成多分辨率点云，添加空值处理和日志"""
        coord = input_dict['coord']
        offset = input_dict['offset']
        batch_size = len(offset)

        # 输入校验：确保offset有效且点云数量合理
        if batch_size <= 0:
            raise ValueError(f"无效的批次大小: {batch_size}，offset应为[B+1]格式")
        if coord.numel() == 0:
            raise ValueError("输入点云坐标为空，无法生成多分辨率点云")

        multi_resolution_dict = []

        for scale in self.scales:
            if scale == 1.0:
                # 原始分辨率：直接复制输入（添加校验）
                if coord.size(0) == 0:
                    raise ValueError("原始点云坐标为空，无法处理")
                res_dict = input_dict.copy()
            else:
                # 生成低分辨率点云：添加详细日志和空值处理
                # 存储每个批次的采样索引，用于同步采样其他特征
                sampled_indices = []  # 新增：保存每个批次的采样索引

                # 第一步：处理坐标并记录采样索引
                sampled_coord = []
                sampled_offset = [0]
                ptr = 0
                batch_size = offset.size(0)
                valid_batch = False  # 标记是否有有效批次

                for i in range(batch_size):
                    end = offset[i]
                    batch_coord = coord[ptr:end]
                    batch_size_i = batch_coord.size(0)

                    #print(f"处理批次 {i + 1}/{batch_size}，原始点数量: {batch_size_i}，缩放比例: {scale}")

                    if batch_size_i == 0:
                        #print(f"警告：批次 {i + 1} 点数量为0，跳过该批次")
                        ptr = end
                        sampled_indices.append(None)  # 空批次索引标记为None
                        continue

                    # 计算采样点数
                    num_samples = max(self.min_samples, int(batch_size_i * scale))
                    num_samples = min(num_samples, batch_size_i)

                    # 采样并记录索引
                    try:
                        if self.sampling_method == 'fps':
                            indices = self._farthest_point_sampling(batch_coord, num_samples)
                        else:  # random
                            indices = torch.randperm(batch_size_i, device=batch_coord.device)[:num_samples]

                        sampled_batch_coord = batch_coord[indices]
                        if sampled_batch_coord.size(0) == 0:
                            raise RuntimeError(f"批次 {i + 1} 采样后点数量为0")

                        sampled_coord.append(sampled_batch_coord)
                        sampled_offset.append(sampled_offset[-1] + sampled_batch_coord.size(0))
                        sampled_indices.append(indices)  # 保存当前批次的采样索引
                        valid_batch = True
                    except Exception as e:
                        # print(f"批次 {i + 1} 采样失败: {str(e)}，使用降级策略")
                        # 降级：取前min_samples个点，记录对应索引
                        indices = torch.arange(min(self.min_samples, batch_size_i), device=batch_coord.device)
                        sampled_batch_coord = batch_coord[indices]
                        sampled_coord.append(sampled_batch_coord)
                        sampled_offset.append(sampled_offset[-1] + sampled_batch_coord.size(0))
                        sampled_indices.append(indices)  # 保存降级策略的索引
                        valid_batch = True

                    ptr = end

                # 处理所有批次都无效的极端情况
                if not valid_batch:
                    # print(f"警告：尺度 {scale} 所有批次采样失败，使用原始点云的前{self.min_samples}个点")
                    fallback_samples = min(self.min_samples, coord.size(0))
                    indices = torch.arange(fallback_samples, device=coord.device)  # 记录降级索引
                    sampled_coord = [coord[indices]]
                    sampled_offset = [0, fallback_samples]
                    sampled_indices = [indices]  # 统一索引格式

                # 拼接坐标和偏移量
                sampled_coord = torch.cat(sampled_coord, dim=0)
                sampled_offset = torch.tensor(sampled_offset, device=coord.device, dtype=offset.dtype)

                # 构建结果字典
                res_dict = {
                    'coord': sampled_coord,
                    'offset': sampled_offset
                }

                # 第二步：处理其他特征（复用已有的采样索引）
                for key in ['feat', 'segment', 'grid_coord']:
                    if key in input_dict and input_dict[key] is not None:
                        sampled_data = []
                        ptr = 0
                        for i in range(batch_size):
                            indices = sampled_indices[i]  # 复用坐标采样时的索引
                            if indices is None:  # 空批次跳过
                                ptr = offset[i]
                                continue

                            end = offset[i]
                            batch_data = input_dict[key][ptr:end]
                            # 直接使用已保存的索引采样特征
                            sampled_batch_data = batch_data[indices]
                            sampled_data.append(sampled_batch_data)
                            ptr = end

                        if sampled_data:
                            res_dict[key] = torch.cat(sampled_data, dim=0)
                        else:
                            # 降级策略：使用原始数据的前几个样本（与坐标保持一致）
                            res_dict[key] = input_dict[key][sampled_indices[0]] if input_dict[key].size(0) > 0 else \
                            input_dict[key]

            multi_resolution_dict.append(res_dict)

        return multi_resolution_dict

    @staticmethod
    def _farthest_point_sampling(xyz, npoint):
        """最远点采样（添加输入校验）"""
        device = xyz.device
        N, C = xyz.shape

        # 输入校验：确保点数量足够
        if N == 0:
            raise ValueError("FPS采样失败：输入点云为空")
        if npoint <= 0:
            raise ValueError(f"FPS采样失败：采样点数必须为正，实际为{npoint}")
        if npoint > N:
            # print(f"警告：FPS采样点数 {npoint} 超过原始点数量 {N}，自动调整为{N}")
            npoint = N

        centroids = torch.zeros(npoint, dtype=torch.long, device=device)
        distance = torch.ones(N, dtype=torch.float32, device=device) * 1e10
        # 初始点选择：避免随机选择时可能的索引错误
        farthest = torch.tensor(0, dtype=torch.long, device=device)  # 固定选择第一个点作为初始点

        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids


@MODELS.register_module()
class MultiResCFPSegmentor(nn.Module):
    """基于多分辨率点云的CFP分割器（修复特征维度不匹配）"""

    def __init__(
            self,
            num_classes,
            backbone_out_channels,  # 配置中设定的backbone输出维度
            backbone=None,
            criteria=None,
            freeze_backbone=False,
            cfp_reduction_ratio=4,
            scales=[1.0, 0.5, 0.25],
            sampling_method='fps',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_out_channels = backbone_out_channels  # 保存配置的维度
        self.actual_backbone_out = None  # 用于动态记录实际输出维度

        # 多分辨率点云生成器
        self.multi_res_generator = MultiResolutionPointGenerator(
            scales=scales,
            sampling_method=sampling_method
        )

        # 共享backbone
        self.backbone = build_model(backbone)

        # CFP融合模块（使用配置的维度初始化）
        self.cfp_fusion = PointCFPFusionModule(
            in_channels=backbone_out_channels,
            reduction_ratio=cfp_reduction_ratio
        )

        # 分割头（临时用占位符，将在第一次前向时动态修正）
        self.seg_head = nn.Linear(backbone_out_channels, num_classes) if num_classes > 0 else nn.Identity()

        # 损失函数
        self.criteria = build_criteria(criteria)

        # 冻结backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        # 1. 生成多分辨率点云（添加异常捕获）
        try:
            multi_res_dicts = self.multi_res_generator(input_dict)
            # print(f"成功生成 {len(multi_res_dicts)} 个分辨率的点云")
        except Exception as e:
            # print(f"多分辨率点云生成失败: {str(e)}，使用原始点云继续")
            multi_res_dicts = [input_dict.copy()]  # 降级为单分辨率

        # 2. 为每个分辨率提取特征
        multi_res_features = []
        multi_res_coords = []
        multi_res_offsets = []

        for i, res_dict in enumerate(multi_res_dicts):
            # 验证当前分辨率的点云是否有效
            if res_dict['coord'].numel() == 0:
                # print(f"警告：分辨率 {i} 点云为空，跳过该分辨率")
                continue

            point = Point(res_dict)
            point = self.backbone(point)

            # 提取最终特征
            if isinstance(point, Point):
                current_point = point
                while "pooling_parent" in current_point.keys():
                    current_point = current_point["pooling_parent"]
                feat = current_point.feat
                coord = current_point.coord
                offset = current_point.offset
            else:
                feat = point
                coord = res_dict["coord"]
                offset = res_dict["offset"]

            # 关键修复：记录实际backbone输出维度并校验
            current_feat_dim = feat.size(1)
            if self.actual_backbone_out is None:
                self.actual_backbone_out = current_feat_dim
                # print(f"检测到实际backbone输出维度: {self.actual_backbone_out}")

                # 如果配置的维度与实际不符，动态修正seg_head和cfp_fusion
                if self.actual_backbone_out != self.backbone_out_channels:
                    # print(
                        # f"警告：配置的backbone_out_channels ({self.backbone_out_channels}) 与实际输出维度 ({self.actual_backbone_out}) 不匹配，自动修正")
                    # 重新初始化分割头
                    self.seg_head = nn.Linear(self.actual_backbone_out, self.num_classes).to(feat.device)
                    # 重新初始化CFP融合模块
                    self.cfp_fusion = PointCFPFusionModule(
                        in_channels=self.actual_backbone_out,
                        reduction_ratio=self.cfp_fusion.reduction_ratio
                    ).to(feat.device)
            else:
                # 确保所有分辨率的特征维度一致
                assert current_feat_dim == self.actual_backbone_out, \
                    f"特征维度不一致：当前 {current_feat_dim} vs 预期 {self.actual_backbone_out}"

            multi_res_features.append(feat)
            multi_res_coords.append(coord)
            multi_res_offsets.append(offset)

            # print(f"分辨率 {i} (缩放比例 {self.multi_res_generator.scales[i]}): "
            #       f"特征数量={feat.size(0)}, 特征维度={current_feat_dim}, 坐标数量={coord.size(0)}")

        # 处理没有有效特征的极端情况
        if not multi_res_features:
            raise RuntimeError("所有分辨率的特征提取失败，无法继续")

        # 3. 从最高分辨率开始融合低分辨率特征
        fused_feat = multi_res_features[0]
        fused_coord = multi_res_coords[0]
        fused_offset = multi_res_offsets[0]

        # 逐层融合低分辨率特征
        for i in range(1, len(multi_res_features)):
            # print(f"融合第 {i} 个分辨率特征 (缩放比例 {self.multi_res_generator.scales[i]})")

            fused_feat = self.cfp_fusion(
                high_res_feat=fused_feat,
                high_res_coord=fused_coord,
                low_res_feat=multi_res_features[i],
                low_res_coord=multi_res_coords[i],
                high_res_offset=fused_offset
            )

        # 4. 上采样到原始分辨率（如果需要）
        original_point_count = input_dict["coord"].size(0)
        if fused_feat.size(0) != original_point_count:
            # print(f"上采样特征从 {fused_feat.size(0)} 到 {original_point_count}")
            fused_feat = self._upsample_to_original(
                fused_feat, fused_coord, input_dict["coord"]
            )

        # 5. 生成分割结果（此时seg_head已确保维度匹配）
        seg_logits = self.seg_head(fused_feat)

        # 6. 计算损失
        return_dict = {}

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            if loss.dim() > 0:
                loss = loss.mean()
            return_dict["loss"] = loss

        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            if loss.dim() > 0:
                loss = loss.mean()
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits




        if return_point:
            return_dict["point"] = Point(input_dict)

        return return_dict
    @staticmethod
    def _upsample_to_original(feat, current_coords, original_coords):
        """上采样到原始点云数量"""
        if current_coords.device != original_coords.device:
            current_coords = current_coords.to(original_coords.device)
            feat = feat.to(original_coords.device)

        # 计算最近邻映射
        dist_matrix = torch.cdist(original_coords.unsqueeze(0), current_coords.unsqueeze(0)).squeeze(0)
        nearest_indices = torch.argmin(dist_matrix, dim=1)

        return feat[nearest_indices]
