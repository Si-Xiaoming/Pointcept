import torch
from pointcept.models import build_model
import os
import time
import numpy as np
from collections import OrderedDict
import pointcept.utils.comm as comm
from torchinfo import summary
from misc.transform import default
MODEL = dict(
    type="DefaultSegmentorV3",
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m3",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=True,

        lora_rank=4,  # LoRA 的秩，None 表示不使用 LoRA
        lora_alpha=16,  # LoRA 缩放因子
        lora_dropout=0.0,  # LoRA dropout
    ),
    freeze_backbone=False,
)

def load_data(data_root):
    coord_np_file = f"{data_root}/coord.npy"
    color_np_file = f"{data_root}/color.npy"
    segment_np_file = f"{data_root}/segment.npy"
    coord = np.load(coord_np_file)
    color = np.load(color_np_file)
    segment = np.load(segment_np_file)

    # concate coord and color and segment as Point
    data_dict = {
        "coord": coord.astype(np.float32),
        "color": color.astype(np.float32),
        "segment": segment.reshape([-1]).astype(np.int32)
    }

    return data_dict
def main_process():
    from pointcept.models.point_transformer_v3.point_transformer_v3m3_lora import PointTransformerV3
    model = PointTransformerV3()
    # data_root = r"/datasets/navarra-test/processed/test/02"
    # data_root = r"/datasets/internship/unused_land_data/processed/test/ground_processed"
    data_root = r"/datasets/navarra-test_only/processed/test/04"
    points = load_data(data_root)
    transform = default()
    points = transform(points)
    summary(model, input_data=points)
main_process()