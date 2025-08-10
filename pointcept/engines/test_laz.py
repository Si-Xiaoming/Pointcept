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
from torch import Tensor
import pdal

try:
    import pointops
except:
    pointops = None


from .test import TESTERS, TesterBase

@TESTERS.register_module()
class SemSegTesterLaz:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False, load_strict=True) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.load_strict = load_strict
        self.verbose = verbose
        if self.verbose and model is None:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config: \n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.point_cloud = self.load_data()
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)

    def build_test_loader(self):
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(self.point_cloud)
        else:
            test_sampler = None

        test_loader = torch.utils.data.DataLoader(
            self.point_cloud,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    @staticmethod
    def normalize_attributes(array: np.array, min_value: float, max_value: float, mean_value: float, std_value: float):
        """根据均值和标准差进行归一化操作，采用2倍标准差归一化

        Args:
            array (np.array): 输入数据
            min_value (float): 初始最小值
            max_value (float): 初始最大值
            mean_value (float): 均值
            std_value (float): 标准差

        Returns:
            object (np.array): 归一化，并且维度变成2维，方便 cat

        """
        min_value = np.clip(mean_value - 2 * std_value, min_value, max_value)
        max_value = np.clip(mean_value + 2 * std_value, min_value, max_value)
        value = (array.astype(np.float32) - min_value) / (max_value - min_value)
        value = np.clip(value, 0.0, 1.0)
        return np.expand_dims(value, 1)

    def load_data(self):
        # laz file dir:
        laz_path = self.cfg.test_laz

        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=laz_path)
        pipeline |= pdal.Filter.stats(dimensions="Intensity,Red,Blue,Green")
        pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=self.cfg.grid_size)
        if self.cfg.has_label:
            pipeline |= pdal.Filter.range(limits="Classification[2:6], Classification[8:8]")
            pipeline |= pdal.Filter.assign(value=[
                f"Classification = 0 WHERE Classification == 2",
                f"Classification = 1 WHERE ((Classification >= 3) && (Classification <= 5))",
                f"Classification = 2 WHERE Classification == 6",
                f"Classification = 3 WHERE Classification == 8"
            ])
        else:
            print("No label, only process point cloud.")
        count = pipeline.execute()
        arrays = pipeline.arrays[0]
        metadata = pipeline.metadata['metadata']
        bounds = metadata['readers.las']
        minx = bounds['minx']
        miny = bounds['miny']
        minz = bounds['minz']
        pos = np.concatenate([
            np.expand_dims(arrays['X'] - minx, 1),
            np.expand_dims(arrays['Y'] - miny, 1),
            np.expand_dims(arrays['Z'] - minz, 1)
        ], axis=-1).astype(np.float32)
        y = arrays['Classification'].reshape(-1, 1)

        stats = metadata['filters.stats']['statistic'][1]
        r = self.normalize_attributes(arrays['Red'], 0.0, (float)(stats['maximum']), stats['average'],
                                      stats['stddev'])
        stats = metadata['filters.stats']['statistic'][2]
        g = self.normalize_attributes(arrays['Green'], 0.0, (float)(stats['maximum']), stats['average'],
                                      stats['stddev'])
        stats = metadata['filters.stats']['statistic'][3]
        b = self.normalize_attributes(arrays['Blue'], 0.0, (float)(stats['maximum']), stats['average'],
                                      stats['stddev'])
        color = np.concatenate([r, g, b], axis=-1)
        data_dict = {}
        data_dict["coord"] = pos.astype(np.float32)
        data_dict["color"] = color.astype(np.float32)
        data_dict["segment"] = y.reshape([-1]).astype(np.int32)
        return data_dict



        

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=self.load_strict)  # True
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process

        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)

