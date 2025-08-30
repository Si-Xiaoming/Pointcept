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
from misc.transform import default
try:
    import flash_attn
except ImportError:
    flash_attn = None


MODEL = model = dict(
    type="Sonata-v1m1",
    # backbone - student & teacher
    backbone=dict(
        type="PT-v3m2",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=1088,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)
WEIGHT = "/datasets/exp/default_lora/dmh/model_last.pth"

def load_model(keywords = "module.student.backbone", replacement = "module.backbone"):
    model = build_model(MODEL)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if os.path.isfile(WEIGHT):
        print(f"Loading weight at: {WEIGHT}")
        checkpoint = torch.load(WEIGHT, weights_only=False)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if keywords in key:
                key = key.replace(keywords, replacement, 1)
            if comm.get_world_size() == 1:
                key = key[7:]  # module.xxx.xxx -> xxx.xxx
            weight[key] = value
        model.load_state_dict(weight, strict=False)  # True
        print(
            "=> Loaded weight '{}' (epoch {})".format(
                WEIGHT, checkpoint["epoch"]
            )
        )
    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(WEIGHT))
    return model

def main_process():
    model = load_model(keywords="", replacement="")
    student_backbone = model.student.backbone
    checkpoint = {
        "state_dict": student_backbone.state_dict(),
    }

    # 保存 backbone 的状态字典
    torch.save(checkpoint, "/datasets/exp/student_backbone.pth")

main_process()