import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)
import pointops
import pdal
from sklearn.neighbors import KDTree
from filelock import FileLock
from torch_geometric.data import Data
from .test import TESTERS, TesterBase
import os


class LAZProcessingMixin:
    """LAZ文件处理的混合类"""
    
    def __init__(self):
        self.pdal_array = None
        self.kdtree = None
        self.center = None
        self.srs = None
        self.minx = self.miny = self.minz = 0
        
    def load_laz_file(self, laz_path, grid_size=0.1, has_label=True):
        """加载laz文件并进行预处理"""
        logger = get_root_logger()
        logger.info(f"Loading LAZ file: {laz_path}")
        
        # 创建pdal管道
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=laz_path)
        pipeline |= pdal.Filter.stats(dimensions="Intensity,Red,Blue,Green")
        pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=grid_size)
        pipeline |= pdal.Filter.outlier(method="statistical", multiplier=3.0)
        if has_label:
            pipeline |= pdal.Filter.range(limits="Classification[2:6], Classification[8:8]")
            pipeline |= pdal.Filter.assign(value=[
                f"Classification = 0 WHERE Classification == 2",
                f"Classification = 1 WHERE ((Classification >= 3) && (Classification <= 5))",
                f"Classification = 2 WHERE Classification == 6",
                f"Classification = 3 WHERE Classification == 8"
            ])
        else:
            print("No label, only process point cloud.")
        
        # 执行管道
        pipeline.execute()
        
        # 获取元数据
        metadata = pipeline.metadata['metadata']
        bounds = metadata['readers.las']
        self.minx = bounds['minx']
        self.miny = bounds['miny']
        self.minz = bounds['minz']
        self.center = torch.tensor([self.minx, self.miny, self.minz], dtype=torch.float32)
        self.srs = metadata['readers.las']['comp_spatialreference']
        
        # 获取处理后的数据
        self.pdal_array = pipeline.arrays[0]
        
        # 创建KDTree用于空间搜索
        pos = np.concatenate([
            np.expand_dims(self.pdal_array['X'] - self.minx, 1),
            np.expand_dims(self.pdal_array['Y'] - self.miny, 1),
            np.expand_dims(self.pdal_array['Z'] - self.minz, 1)
        ], axis=-1).astype(np.float32)        
        
        self.kdtree = KDTree(pos)
        
        logger.info(f"Loaded {len(pos)} points from LAZ file")
        return pos
        
    def get_spatial_block(self, center_point, num_points=60000):
        """获取空间上相邻的点块"""
        dists, indices = self.kdtree.query(center_point, k=num_points)
        indices = indices[0]
        # dists = dists[0]
        return indices
    
    def save_predictions_to_laz(self, predictions, output_path):
        """保存预测结果到laz文件"""
        logger = get_root_logger()
        logger.info(f"Saving predictions to LAZ file: {output_path}")
        
        # 创建输出数组的副本
        output_array = self.pdal_array.copy()
        
        # 将预测结果写入Classification字段
        output_array['Classification'] = predictions.astype(np.int32)
        
        # 创建pdal管道保存文件
        pipeline = pdal.Pipeline(arrays=[output_array])
        las_kwargs = {'a_srs': self.srs} if self.srs else {}
        
        pipeline |= pdal.Writer.las(
            filename=output_path,
            minor_version=4,
            scale_x=0.001,
            scale_y=0.001,
            scale_z=0.001,
            offset_x='auto',
            offset_y='auto',
            offset_z='auto',** las_kwargs
        )
        
        pipeline.execute()
        logger.info(f"Successfully saved predictions to {output_path}")

class BlockProcessingMixin:
    """分块处理的混合类"""
    
    def __init__(self):
        self.processed_points = set()
        self.all_predictions = None
        self.point_weights = None
        self.num_points = 0
        self.num_classes = 0
        
    def initialize_processing(self, num_points, num_classes):
        """初始化处理状态"""
        self.num_points = num_points
        self.num_classes = num_classes
        self.all_predictions = torch.zeros((num_points, num_classes), dtype=torch.float32)
        self.point_weights = torch.zeros(num_points, dtype=torch.float32)
        self.processed_points = set()
        
    def process_block(self, block_indices, block_predictions, weight=1.0):
        """处理一个点块并更新全局预测"""
        # 将预测结果累加到全局预测中
        self.all_predictions[block_indices] += block_predictions * weight
        self.point_weights[block_indices] += weight
        
        # 标记这些点为已处理
        self.processed_points.update(block_indices)
        
    def get_unprocessed_regions(self, sample_spacing=5.0):
        """获取未处理的区域中心"""
        if len(self.processed_points) == self.num_points:
            return None
            
        unprocessed_mask = torch.ones(self.num_points, dtype=torch.bool)
        unprocessed_mask[list(self.processed_points)] = False
        unprocessed_indices = torch.where(unprocessed_mask)[0]
        
        if len(unprocessed_indices) == 0:
            return None
            
        # 对未处理的点进行均匀采样作为中心
        sample_indices = unprocessed_indices[::max(1, len(unprocessed_indices) // 20)]
        return sample_indices
        
    def get_final_predictions(self):
        """获取最终的预测结果"""
        # 处理权重为0的点（可能是未被任何块覆盖的点）
        zero_weight_mask = self.point_weights == 0
        if torch.any(zero_weight_mask):
            # 为这些点分配默认类别（背景）
            self.all_predictions[zero_weight_mask, 0] = 1.0
            self.point_weights[zero_weight_mask] = 1.0
            
        # 归一化预测概率
        normalized_preds = self.all_predictions / self.point_weights.unsqueeze(1)
        
        # 获取最终的类别预测
        final_preds = torch.argmax(normalized_preds, dim=1)
        return final_preds.cpu().numpy()



@TESTERS.register_module()
class LAZSemiSegTester(TesterBase, LAZProcessingMixin, BlockProcessingMixin):
    """支持LAZ文件分块推理的语义分割测试器"""
    
    def __init__(self, cfg, model=None, test_loader=None, verbose=False, load_strict=True) -> None:
        TesterBase.__init__(self, cfg, model, test_loader, verbose, load_strict)
        LAZProcessingMixin.__init__(self)
        BlockProcessingMixin.__init__(self)
        
        # 分块处理参数
        self.block_size = cfg.get('block_size', 60000)  # 每个块的点数
        self.grid_size = cfg.get('grid_size', 0.1)      # 预处理网格大小
        self.num_iterations = cfg.get('num_iterations', 5)  # 迭代次数确保覆盖所有点
        
        
    def build_test_loader(self):
        """重写构建测试加载器的方法"""
        # 对于LAZ文件，我们不需要传统的DataLoader
        return None
        
    def test(self):
        """执行LAZ文件的分块推理"""
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start LAZ Evaluation >>>>>>>>>>>>>>>>")
        
        # 检查是否指定了LAZ文件路径
        if not hasattr(self.cfg.data.test, 'laz_path') or not self.cfg.data.test.laz_path:
            raise ValueError("Please specify 'laz_path' in test configuration")
        
        # 加载LAZ文件
        pos = self.load_laz_file(
            self.cfg.data.test.laz_path,
            grid_size=self.grid_size
        )
        
        # 初始化处理状态
        self.initialize_processing(
            num_points=len(pos),
            num_classes=self.cfg.data.num_classes
        )
        
        # 获取原始标签（如果有）
        original_labels = None
        if 'Classification' in self.pdal_array.dtype.fields:
            original_labels = self.pdal_array['Classification'].astype(np.int32)
            logger.info(f"Found original labels with {len(np.unique(original_labels))} classes")
        
        # 开始分块处理
        start_time = time.time()
        
        for iteration in range(self.num_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{self.num_iterations} ===")
            
            # 获取未处理区域的中心
            unprocessed_centers = self.get_unprocessed_regions()
            
            if unprocessed_centers is None:
                logger.info("All points have been processed")
                break
                
            logger.info(f"Found {len(unprocessed_centers)} unprocessed regions to sample")
            
            # 处理每个中心区域
            for center_idx in unprocessed_centers:
                # 获取中心点坐标
                center_point = pos[center_idx:center_idx+1]
                
                # 获取空间相邻的点块
                block_indices = self.get_spatial_block(
                    center_point,
                    num_points=self.block_size,
                )
                
                if len(block_indices) < 100:
                    continue
                    
                logger.info(f"Processing block with {len(block_indices)} points")
                
                # 准备输入数据
                block_pos = pos[block_indices]
                block_feats = self._get_features(block_indices)
                
                # 创建输入字典
                input_dict = {
                    'pos': torch.tensor(block_pos, dtype=torch.float32).cuda(),
                    'x': torch.tensor(block_feats, dtype=torch.float32).cuda(),
                    'batch': torch.zeros(len(block_indices), dtype=torch.int64).cuda(),
                    'offset': torch.tensor([0, len(block_indices)], dtype=torch.int64).cuda()
                }
                input_dict = {
                    "coord": block_pos.astype(np.float32),
                    "color": block_feats.astype(np.float32),
                    # "segment": segment.reshape([-1]).astype(np.int32)
                }
                
                # 模型推理
                with torch.no_grad():
                    output = self.model(input_dict)
                    seg_logits = output['seg_logits']
                    block_preds = F.softmax(seg_logits, dim=1).cpu()
                
                # 更新全局预测
                self.process_block(block_indices, block_preds)
                
                # 记录进度
                processed_ratio = len(self.processed_points) / self.num_points * 100
                logger.info(f"Processed {len(self.processed_points)}/{self.num_points} points ({processed_ratio:.1f}%)")
                
                # 释放GPU内存
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
        
        # 获取最终预测结果
        final_preds = self.get_final_predictions()
        
        # 计算精度指标（如果有原始标签）
        if original_labels is not None:
            self._calculate_metrics(final_preds, original_labels)
        
        # 保存预测结果到LAZ文件
        output_laz_path = self.cfg.get('output_laz_path', 
                                     os.path.join(self.cfg.save_path, 'predictions.laz'))
        self.save_predictions_to_laz(final_preds, output_laz_path)
        
        total_time = time.time() - start_time
        logger.info(f"\nTotal processing time: {total_time:.2f} seconds")
        logger.info(f"Processed {self.num_points} points")
        logger.info(f"Output saved to: {output_laz_path}")
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        
        return final_preds
        
    def _get_features(self, indices):
        """获取点云特征"""  
        
        # 提取颜色特征（如果有）
        color_features = []
        if 'Red' in self.pdal_array.dtype.fields:
            red = self.pdal_array['Red'][indices].astype(np.float32) / 255.0
            green = self.pdal_array['Green'][indices].astype(np.float32) / 255.0
            blue = self.pdal_array['Blue'][indices].astype(np.float32) / 255.0
            color_features = [red, green, blue]
       
        return np.stack(color_features, axis=1)
        
    def _calculate_metrics(self, predictions, labels):
        """计算精度指标"""
        logger = get_root_logger()
        
        # 过滤掉忽略的类别
        ignore_index = self.cfg.data.get('ignore_index', -1)
        valid_mask = labels != ignore_index
        
        if np.sum(valid_mask) == 0:
            logger.warning("No valid labels for evaluation")
            return
            
        valid_preds = predictions[valid_mask]
        valid_labels = labels[valid_mask]
        
        # 计算交并比和准确率
        intersection, union, target = intersection_and_union(
            valid_preds, valid_labels, 
            self.cfg.data.num_classes, 
            ignore_index
        )
        
        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection) / (sum(target) + 1e-10)
        
        logger.info("=" * 50)
        logger.info("Evaluation Metrics")
        logger.info("=" * 50)
        logger.info(f"mIoU: {mIoU:.4f}")
        logger.info(f"mAcc: {mAcc:.4f}")
        logger.info(f"allAcc: {allAcc:.4f}")
        logger.info("-" * 50)
        
        # 打印每个类别的指标
        for i in range(self.cfg.data.num_classes):
            if hasattr(self.cfg.data, 'names') and i < len(self.cfg.data.names):
                class_name = self.cfg.data.names[i]
            else:
                class_name = f"Class_{i}"
            
            logger.info(f"{class_name}: IoU={iou_class[i]:.4f}, Acc={accuracy_class[i]:.4f}")
        
        logger.info("=" * 50)