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

import pointops
import pdal
from sklearn.neighbors import KDTree
from pointcept.engines.test import TesterBase, TESTERS


@TESTERS.register_module()
class LAZSemiSegTesterSimple(TesterBase):
    """
    修复版LAZ语义分割测试器
    关键修复：正确处理transform前后的索引映射
    """

    def __init__(self, cfg, model=None, test_loader=None, verbose=False, load_strict=True) -> None:
        super().__init__(cfg, model, test_loader, verbose, load_strict)
        self.num_points_per_block = cfg.get('num_points_per_block', 60000)
        self.overlap_ratio = cfg.get('overlap_ratio', 0.1)
        self.grid_size = cfg.get('grid_size', 0.1)
        self.vote_rounds = cfg.get('vote_rounds', 3)

        self.is_laz_dataset = False
        if hasattr(self.test_loader.dataset, '__class__'):
            self.is_laz_dataset = self.test_loader.dataset.__class__.__name__ == 'LAZDatasetVote'

        if self.is_laz_dataset:
            self.logger.info("Detected LAZDataset, enabling LAZ-specific processing")
            self.logger.info(f"Using voting mechanism with {self.vote_rounds} rounds")

    def majority_voting(self, predictions):
        """多数投票函数"""
        if len(predictions) == 1:
            return predictions[0]

        pred_labels = np.zeros(predictions[0].shape[0], dtype=np.int32)
        for i in range(predictions[0].shape[0]):
            votes = [pred[i] for pred in predictions]
            pred_labels[i] = np.argmax(np.bincount(votes))

        return pred_labels

    def merge_predictions(self, predictions_dict, dataset):
        """
        合并所有块的预测结果 - 修复版
        关键修复：使用transform_mapping正确映射回原始数据空间
        """
        self.logger.info("Merging predictions from all blocks...")

        # **使用原始数据大小（transform之前）**
        if hasattr(dataset, 'num_total_points'):
            num_points = dataset.num_total_points
            self.logger.info(f"Using original data size: {num_points} points")
        elif hasattr(dataset, 'original_coord') and dataset.original_coord is not None:
            num_points = dataset.original_coord.shape[0]
            self.logger.info(f"Using original_coord size: {num_points} points")
        elif hasattr(dataset, 'current_arrays'):
            num_points = len(dataset.current_arrays)
            self.logger.info(f"Using current_arrays size: {num_points} points")
        else:
            raise RuntimeError("Dataset does not have valid size information")

        num_classes = self.cfg.data.num_classes

        # 初始化概率数组（使用原始大小）
        pred_probs = np.zeros((num_points, num_classes), dtype=np.float32)
        pred_counts = np.zeros(num_points, dtype=np.int32)

        # **检查是否有transform映射**
        has_mapping = hasattr(dataset, 'transform_mapping') and dataset.transform_mapping is not None
        if has_mapping:
            self.logger.info(f"Using transform mapping: {len(dataset.transform_mapping)} processed -> original")
        else:
            self.logger.warning("No transform mapping found, assuming identity mapping")

        # 合并所有块的预测
        for block_info, (pred, indices) in predictions_dict.items():
            # indices是基于processed_coord的索引
            # 需要映射回original_coord的索引

            if has_mapping:
                # **关键修复：将processed索引映射回original索引**
                original_indices = dataset.transform_mapping[indices.cpu().numpy()]
            else:
                # 如果没有映射，直接使用索引（假设没有transform）
                original_indices = indices.cpu().numpy()

            # 确保索引在有效范围内
            valid_mask = original_indices < num_points
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                self.logger.warning(f"Found {invalid_count} invalid indices in block {block_info}, filtering...")
                original_indices = original_indices[valid_mask]
                pred = pred[valid_mask]

            if len(original_indices) == 0:
                self.logger.warning(f"Block {block_info} has no valid points, skipping...")
                continue

            pred_np = pred.data.cpu().numpy()

            # 确保形状匹配
            if pred_np.ndim == 1:
                pred_prob_np = np.zeros((len(pred_np), num_classes), dtype=np.float32)
                for i, label in enumerate(pred_np):
                    if 0 <= label < num_classes:
                        pred_prob_np[i, label] = 1.0
            else:
                pred_prob_np = pred_np

            # 使用original_indices更新预测
            pred_probs[original_indices] += pred_prob_np
            pred_counts[original_indices] += 1

        # 处理未预测的点
        mask = pred_counts == 0
        if np.any(mask):
            self.logger.warning(f"Found {np.sum(mask)} points without predictions, assigning to default class")
            pred_probs[mask, 0] = 1.0
            pred_counts[mask] = 1

        # 平均概率
        pred_probs = pred_probs / pred_counts[:, np.newaxis]

        # 获取最终预测
        pred_labels = np.argmax(pred_probs, axis=1)

        self.logger.info(f"Merged predictions: {len(pred_labels)} points")
        self.logger.info(f"Coverage: {np.sum(pred_counts > 0)} / {num_points} points have predictions")

        return pred_labels, pred_probs

    def save_result_laz(self, dataset, pred_labels, output_path):
        """保存预测结果为LAZ文件"""
        self.logger.info(f"Saving prediction result to: {output_path}")

        if not hasattr(dataset, 'current_arrays'):
            raise RuntimeError("Dataset does not have current_arrays attribute")

        # 验证大小匹配
        if len(pred_labels) != len(dataset.current_arrays):
            raise ValueError(
                f"Prediction size mismatch! pred_labels: {len(pred_labels)}, "
                f"current_arrays: {len(dataset.current_arrays)}"
            )

        # 创建新的PDAL数组
        new_array = dataset.current_arrays.copy()

        # 更新分类标签
        new_array['Classification'] = pred_labels.astype(np.int32)

        # 创建PDAL管道保存文件
        pipeline = pdal.Pipeline(arrays=[new_array])

        las_kwargs = {
            'minor_version': 4,
            'scale_x': 0.001,
            'scale_y': 0.001,
            'scale_z': 0.001,
            'offset_x': 'auto',
            'offset_y': 'auto',
            'offset_z': 'auto'
        }

        if hasattr(dataset, 'current_metadata') and 'readers.las' in dataset.current_metadata:
            if 'comp_spatialreference' in dataset.current_metadata['readers.las']:
                las_kwargs['a_srs'] = dataset.current_metadata['readers.las']['comp_spatialreference']

        pipeline |= pdal.Writer.las(filename=output_path, **las_kwargs)
        pipeline.execute()

        self.logger.info("Successfully saved LAZ file with predictions")

    def test(self):
        """测试方法 - 修复版"""
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start LAZ Evaluation (Fixed Version) >>>>>>>>>>>>>>>>")
        logger.info(f"Using voting mechanism with {self.vote_rounds} rounds")

        if not self.is_laz_dataset:
            logger.warning("LAZSemiSegTesterSimple is designed for LAZDataset")
            return super().test()

        batch_time = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)

        predictions_dict = {}
        dataset = self.test_loader.dataset
        num_blocks = len(dataset)
        logger.info(f"Total blocks to process: {num_blocks}")

        # 处理每个块
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()

            if isinstance(data_dict, list) and len(data_dict) > 0:
                data_dict = data_dict[0]

            block_idx = data_dict.get("block_idx", idx)
            num_blocks_total = data_dict.get("num_blocks", num_blocks)
            data_name = data_dict.get("name", f"block_{block_idx}")

            # 获取块的索引（基于processed_coord）
            block_indices = data_dict['data'].get("index", None)
            if block_indices is None:
                self.logger.warning(f"No index found in block {block_idx}, skipping...")
                continue

            # 验证索引有效性（基于processed数据大小）
            if hasattr(dataset, 'processed_coord') and dataset.processed_coord is not None:
                data_size = dataset.processed_coord.shape[0]
            elif hasattr(dataset, 'loaded_data') and dataset.loaded_data is not None:
                data_size = dataset.loaded_data['coord'].shape[0]
            else:
                self.logger.warning(f"Cannot determine processed data size for block {block_idx}")
                continue

            valid_mask = block_indices < data_size
            if not torch.all(valid_mask):
                invalid_count = torch.sum(~valid_mask)
                self.logger.warning(f"Found {invalid_count} invalid indices in block {block_idx}, filtering...")
                block_indices = block_indices[valid_mask]

            if len(block_indices) == 0:
                self.logger.warning(f"Block {block_idx} has no valid points, skipping...")
                continue

            # 初始化投票计数器
            vote_results = []

            # 进行多轮投票
            for vote_round in range(self.vote_rounds):
                # 准备输入数据
                input_dict = {}
                for key in data_dict['data'].keys():
                    if data_dict['data'][key] is not None:
                        if isinstance(data_dict['data'][key], np.ndarray):
                            input_dict[key] = torch.tensor(data_dict['data'][key]).cuda(non_blocking=True)
                        elif isinstance(data_dict['data'][key], torch.Tensor):
                            input_dict[key] = data_dict['data'][key].cuda(non_blocking=True)

                # 模型推理
                with torch.no_grad():
                    try:
                        pred = self.model(input_dict)["seg_logits"]
                        pred_label = pred.max(1)[1].data.cpu().numpy()
                        vote_results.append(pred_label)

                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()

                    except Exception as e:
                        self.logger.error(f"Error in vote round {vote_round} for block {block_idx}: {str(e)}")
                        continue

            # 检查投票结果
            if len(vote_results) == 0:
                self.logger.warning(f"No valid vote results for block {block_idx}, skipping...")
                continue

            # 进行多数投票
            final_pred = self.majority_voting(vote_results)

            # 存储块预测结果（使用概率形式）
            pred_probs = np.zeros((len(final_pred), self.cfg.data.num_classes), dtype=np.float32)
            for i, label in enumerate(final_pred):
                if 0 <= label < self.cfg.data.num_classes:
                    pred_probs[i, label] = 1.0

            # **关键：block_indices是基于processed_coord的索引**
            predictions_dict[block_idx] = (torch.tensor(pred_probs).cuda(), block_indices)

            batch_time.update(time.time() - start)

            logger.info(
                "Block {}/{}: {} points, {} votes, Time: {:.3f}s, Avg Time: {:.3f}s".format(
                    block_idx + 1, num_blocks_total, len(block_indices),
                    len(vote_results), batch_time.val, batch_time.avg
                )
            )

        # 合并所有块的预测结果
        if len(predictions_dict) == 0:
            self.logger.error("No valid blocks processed!")
            return None

        # **关键：merge_predictions会使用transform_mapping映射回原始空间**
        pred_labels, pred_probs = self.merge_predictions(predictions_dict, dataset)

        # 计算整体精度（使用原始标签）
        if hasattr(dataset, 'original_segment') and dataset.original_segment is not None:
            logger.info("Calculating overall accuracy metrics...")

            segment_data = dataset.original_segment

            # 验证大小匹配
            if len(pred_labels) != len(segment_data):
                logger.error(
                    f"Size mismatch! pred_labels: {len(pred_labels)}, "
                    f"segment_data: {len(segment_data)}"
                )
            else:
                # 过滤忽略标签
                valid_mask = segment_data != self.cfg.data.ignore_index
                if np.any(valid_mask):
                    pred_labels_valid = pred_labels[valid_mask]
                    segment_valid = segment_data[valid_mask]

                    intersection, union, target = intersection_and_union(
                        pred_labels_valid, segment_valid,
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
                else:
                    logger.warning("No valid labels found for accuracy calculation")
        elif hasattr(dataset, 'current_segment') and dataset.current_segment is not None:
            logger.warning("Using current_segment instead of original_segment for accuracy calculation")
            segment_data = dataset.current_segment

            # 同样的精度计算逻辑...
            valid_mask = segment_data != self.cfg.data.ignore_index
            if np.any(valid_mask):
                pred_labels_valid = pred_labels[valid_mask]
                segment_valid = segment_data[valid_mask]

                intersection, union, target = intersection_and_union(
                    pred_labels_valid, segment_valid,
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
        else:
            logger.warning("No ground truth segment found in dataset")
            mIoU = None
            mAcc = None
            allAcc = None

        # 保存结果
        output_laz_path = os.path.join(save_path, "predictions.laz")
        self.save_result_laz(dataset, pred_labels, output_laz_path)

        np.save(os.path.join(save_path, "pred_labels.npy"), pred_labels)
        np.save(os.path.join(save_path, "pred_probs.npy"), pred_probs)

        if hasattr(dataset, 'original_segment') and dataset.original_segment is not None:
            np.save(os.path.join(save_path, "ground_truth.npy"), dataset.original_segment)

        logger.info("<<<<<<<<<<<<<<<<< End LAZ Evaluation <<<<<<<<<<<<<<<<<")

        return {
            'pred_labels': pred_labels,
            'pred_probs': pred_probs,
            'output_laz_path': output_laz_path,
            'num_points': dataset.num_total_points if hasattr(dataset, 'num_total_points') else len(pred_labels),
            'num_blocks': num_blocks,
            'vote_rounds': self.vote_rounds,
            'mIoU': mIoU if 'mIoU' in locals() else None,
            'mAcc': mAcc if 'mAcc' in locals() else None,
            'allAcc': allAcc if 'allAcc' in locals() else None
        }

    @staticmethod
    def collate_fn(batch):
        return batch