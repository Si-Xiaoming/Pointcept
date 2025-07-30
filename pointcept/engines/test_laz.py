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
class EvaluateSSLModel(TesterBase):
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
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
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

    @staticmethod
    def collate_fn(batch):
        return batch