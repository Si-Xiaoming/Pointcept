import torch
import numpy as np
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
import open3d as o3d
import os
import time
import numpy as np
from collections import OrderedDict
from pointcept.datasets import NavarraDataset
from pointcept.models.utils.structure import Point
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence
from .transform import default

try:
    import flash_attn
except ImportError:
    flash_attn = None

from misc import load_model, load_data

def main_process():
    data_root = r"/datasets/navarra-test/processed/test/02"
    points = load_data(data_root)
    transform = default()
    points = transform(points)
    for key in points.keys():
        if isinstance(points[key], torch.Tensor):
            points[key] = points[key].cuda(non_blocking=True)

    model = load_model(keywords="", replacement="")
    model = model.cuda()
    model.eval()  # 确保模型在评估模式
    traced_model = torch.jit.trace(model, points, strict=False)

    # 保存模型
    traced_model.save("traced_model.pt")

if __name__ == "__main__":
    main_process()
