import torch
from pointcept.models import build_model
import os
import time
import numpy as np
from collections import OrderedDict
import pointcept.utils.comm as comm



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
        freeze_encoder=False,
        lora_rank=4,  # LoRA 的秩，None 表示不使用 LoRA
        lora_alpha=16,  # LoRA 缩放因子
        lora_dropout=0.0,  # LoRA dropout
    ),
)
WEIGHT = '/datasets/exp/default_lora/model/model_last.pth'
def load_model(keywords="module.student.backbone", replacement="module.backbone"):
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
        print(f"Loaded {n_parameters}")
        print()
        print(
            "=> Loaded weight '{}' (epoch {})".format(
                WEIGHT, checkpoint["epoch"]
            )
        )
    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(WEIGHT))
    return model

if __name__ == '__main__':
    model = load_model()
    print(model)
    print(model.state_dict().keys())