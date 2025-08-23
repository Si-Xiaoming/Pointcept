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

MODEL = dict(
    type="DefaultSegmentorV2",
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
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
    ),

)


# MODEL = dict(
#     type="DefaultSegmentorV2",
#     num_classes=4,
#     backbone_out_channels=1232,   # 1232
#     backbone=dict(
#         type="PT-v3m2",
#         in_channels=6,
#         order=("z", "z-trans", "hilbert", "hilbert-trans"),
#         stride=(2, 2, 2, 2),
#         enc_depths=(3, 3, 3, 12, 3),
#         enc_channels=(48, 96, 192, 384, 512),
#         enc_num_head=(3, 6, 12, 24, 32),
#         enc_patch_size=(1024, 1024, 1024, 1024, 1024),
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.3,
#         shuffle_orders=True,
#         pre_norm=True,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=False,
#         upcast_softmax=False,
#         traceable=False,
#         mask_token=False,
#         enc_mode=True,
#         freeze_encoder=False,
#     ),
#
#     freeze_backbone=True,
# )


# WEIGHT = "/datasets/exp/model/model_last-ep9.pth"
WEIGHT = "/datasets/exp/model_best.pth"

def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color



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
    # data_root = r"/datasets/navarra-test/processed/test/02"
    # data_root = r"/datasets/internship/unused_land_data/processed/test/ground_processed"
    data_root = r"/datasets/navarra-test_only/processed/test/04"
    points = load_data(data_root)
    transform = default()
    points = transform(points)

    model = load_model(keywords="", replacement="")
    model = model.cuda()
    with torch.inference_mode():
        for key in points.keys():
            if isinstance(points[key], torch.Tensor):
                points[key] = points[key].cuda(non_blocking=True)
        # model forward:
        points= model(points, return_point=True)
        
        pca_color = get_pca_color(points["point"]["feat"], brightness=1.2, center=True)
    # original_pca_color = pca_color[points.inverse]

    # convert  points["point"]["feat"] to numpy
    feat = points["point"]["feat"].cpu().numpy()
    
        # save original_coord and original_pca_color to numpy files
    np.save("/datasets/vis/coord.npy", points["point"]["coord"].cpu().numpy())
    np.save("/datasets/vis/segment.npy", points["point"]["segment"].cpu().numpy())
    np.save("/datasets/vis/color.npy", pca_color.cpu().numpy())
    np.save("/datasets/vis/feat.npy", feat)

if __name__ == "__main__":
    main_process()
