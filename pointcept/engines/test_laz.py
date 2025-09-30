import json
from uuid import uuid4
import os
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

try:
    import pointops
except:
    pointops = None

# 添加PDAL依赖

import pdal
from sklearn.neighbors import KDTree
from pointcept.engines.test import TesterBase,TESTERS


@TESTERS.register_module()
class LAZSemiSegTester(TesterBase):
    """
    支持直接处理LAZ文件的语义分割测试器
    特点：
    1. 直接读取LAZ文件，无需格式转换
    2. 空间分块加载，每次处理6万个相邻点
    3. 确保所有点都被处理
    4. 输出精度评价和预测后的LAZ文件
    """

    def __init__(self, cfg, model=None, test_loader=None, verbose=False, load_strict=True) -> None:
        super().__init__(cfg, model, test_loader, verbose, load_strict)
        self.num_points_per_block = cfg.get('num_points_per_block', 60000)
        self.overlap_ratio = cfg.get('overlap_ratio', 0.1)
        self.grid_size = cfg.get('grid_size', 0.1)

        # 检查是否是LAZ数据集
        self.is_laz_dataset = isinstance(self.test_loader.dataset, type('LAZDataset', (), {})) or \
                              (hasattr(self.test_loader.dataset, '__class__') and
                               self.test_loader.dataset.__class__.__name__ == 'LAZDataset')

        if self.is_laz_dataset:
            self.logger.info("Detected LAZDataset, enabling LAZ-specific processing")

    def merge_predictions(self, predictions_dict, dataset):
        """合并所有块的预测结果"""
        self.logger.info("Merging predictions from all blocks...")

        # 获取完整点云信息
        full_data = dataset.get_data(0)  # LAZDataset通常只有一个数据项
        num_points = full_data['coord'].shape[0]
        num_classes = self.cfg.data.num_classes

        # 初始化概率数组
        pred_probs = np.zeros((num_points, num_classes), dtype=np.float32)
        pred_counts = np.zeros(num_points, dtype=np.int32)

        # 合并所有块的预测
        for block_idx, (pred, indices) in predictions_dict.items():
            pred_np = pred.data.cpu().numpy()
            pred_probs[indices] += pred_np
            pred_counts[indices] += 1

        # 处理未预测的点
        mask = pred_counts == 0
        if np.any(mask):
            self.logger.warning(f"Found {np.sum(mask)} points without predictions, assigning to default class")
            pred_probs[mask, 0] = 1.0  # 分配给第一个类
            pred_counts[mask] = 1

        # 平均概率
        pred_probs = pred_probs / pred_counts[:, np.newaxis]

        # 获取最终预测
        pred_labels = np.argmax(pred_probs, axis=1)

        return pred_labels, pred_probs, full_data

    def save_result_laz(self, full_data, pred_labels, output_path):
        """保存预测结果为LAZ文件"""
        self.logger.info(f"Saving prediction result to: {output_path}")

        # 创建新的PDAL数组
        new_array = full_data['original_arrays'].copy()

        # 更新分类标签
        new_array['Classification'] = pred_labels.astype(np.int32)

        # 创建PDAL管道保存文件
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
        if 'original_metadata' in full_data and 'comp_spatialreference' in full_data['original_metadata'][
            'readers.las']:
            las_kwargs['a_srs'] = full_data['original_metadata']['readers.las']['comp_spatialreference']

        pipeline |= pdal.Writer.las(filename=output_path, **las_kwargs)
        pipeline.execute()

        self.logger.info("Successfully saved LAZ file with predictions")

    def test(self):
        """主要测试方法"""
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start LAZ Evaluation >>>>>>>>>>>>>>>>")

        if not self.is_laz_dataset:
            logger.warning("LAZSemiSegTester is designed for LAZDataset, falling back to default testing")
            return super().test()

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)

        # 存储所有块的预测结果
        predictions_dict = {}

        # 获取数据集信息
        dataset = self.test_loader.dataset
        num_blocks = len(dataset)
        logger.info(f"Total blocks to process: {num_blocks}")

        # 处理每个块
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()

            # 由于我们使用了collate_fn，需要处理batch维度
            if isinstance(data_dict, list) and len(data_dict) > 0:
                data_dict = data_dict[0]

            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            block_idx = data_dict.get("block_idx", idx)
            num_blocks_total = data_dict.get("num_blocks", num_blocks)

            # 获取块索引
            block_indices = None
            if "index" in data_dict:
                block_indices = data_dict["index"]
            elif len(fragment_list) > 0 and "index" in fragment_list[0]:
                block_indices = fragment_list[0]["index"]

            if block_indices is None:
                raise RuntimeError("Block indices not found in data")

            valid_mask = block_indices < len(dataset.current_arrays)
            if not torch.all(valid_mask):
                invalid_count = torch.sum(~valid_mask)
                self.logger.warning(f"Found {invalid_count} invalid indices in block {block_idx}, filtering...")
                block_indices = block_indices[valid_mask]

            if len(block_indices) == 0:
                self.logger.warning(f"Block {block_idx} has no valid points, skipping...")
                continue

            # 处理块推理
            pred = torch.zeros((len(block_indices), self.cfg.data.num_classes)).cuda()

            for i in range(len(fragment_list)):
                input_dict = fragment_list[i]

                # 转换为CUDA张量
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)

                with torch.no_grad():
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)

                    if "index" in input_dict:
                        idx_part = input_dict["index"]
                        pred[idx_part, :] += pred_part
                    else:
                        pred += pred_part

                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()

            # 存储块预测结果
            predictions_dict[block_idx] = (pred, block_indices)

            # 计算当前块的精度（如果有标签）
            if segment is not None and len(segment) > 0:
                current_pred = pred.max(1)[1].data.cpu().numpy()
                current_segment = segment.data.cpu().numpy() if isinstance(segment, torch.Tensor) else segment

                intersection, union, target = intersection_and_union(
                    current_pred, current_segment,
                    self.cfg.data.num_classes, self.cfg.data.ignore_index
                )

                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)

            batch_time.update(time.time() - start)

            logger.info(
                "Block {}/{}: {} points, Time: {:.3f}s, Avg Time: {:.3f}s".format(
                    block_idx + 1, num_blocks_total, len(block_indices),
                    batch_time.val, batch_time.avg
                )
            )

        # 合并所有块的预测结果
        pred_labels, pred_probs, full_data = self.merge_predictions(predictions_dict, dataset)

        # 计算整体精度（如果有标签）
        if full_data.get('segment') is not None:
            logger.info("Calculating overall accuracy metrics...")

            intersection, union, target = intersection_and_union(
                pred_labels, full_data['segment'],
                self.cfg.data.num_classes, self.cfg.data.ignore_index
            )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Evaluation Result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )

            for i in range(self.cfg.data.num_classes):
                class_name = self.cfg.data.names[i] if hasattr(self.cfg.data, 'names') and i < len(
                    self.cfg.data.names) else f"Class_{i}"
                logger.info(
                    "Class_{idx} - {name}: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=class_name,
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )

        # 保存结果
        # 保存预测结果LAZ文件
        output_laz_path = os.path.join(save_path, "predictions.laz")
        self.save_result_laz(full_data, pred_labels, output_laz_path)

        # 保存预测概率和标签
        np.save(os.path.join(save_path, "pred_labels.npy"), pred_labels)
        np.save(os.path.join(save_path, "pred_probs.npy"), pred_probs)

        if full_data.get('segment') is not None:
            np.save(os.path.join(save_path, "ground_truth.npy"), full_data['segment'])

        logger.info("<<<<<<<<<<<<<<<<< End LAZ Evaluation <<<<<<<<<<<<<<<<<")

        return {
            'pred_labels': pred_labels,
            'pred_probs': pred_probs,
            'output_laz_path': output_laz_path,
            'num_points': full_data['coord'].shape[0],
            'num_blocks': num_blocks,
            'mIoU': mIoU if 'mIoU' in locals() else None,
            'mAcc': mAcc if 'mAcc' in locals() else None,
            'allAcc': allAcc if 'allAcc' in locals() else None
        }

    @staticmethod
    def collate_fn(batch):
        return batch