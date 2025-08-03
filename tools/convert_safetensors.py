import argparse
import os
import numpy as np
import torch.jit
import torch
from safetensors.torch import save_file
from vis.transform import default
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.models.utils.structure import Point  # 确保导入 Point 类
import pointcept.utils.comm as comm
from collections import OrderedDict
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
WEIGHT = "/datasets/exp-0801/model/model_best-centerloss.pth"

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
        print(
            "=> Loaded weight '{}' (epoch {})".format(
                WEIGHT, checkpoint["epoch"]
            )
        )
    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(WEIGHT))
    return model

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

    # 如何将model转为torchscript的格式
    model.eval()  # 非常重要！确保 BatchNorm, Dropout 等层处于推理模式
    # 1. 准备示例输入
    # `points` 字典已经是模型期望的格式和设备 (GPU)
    # 我们直接使用它作为示例输入
    example_input_dict = points  # 这个字典包含 'coord', 'color', 'segment'
    example_input = (
        example_input_dict,  # 对应 forward 方法的 input_dict 参数
        # False # 如果需要传递 return_point 参数，可以在这里添加
    )

    dummy_input = example_input_dict  # 使用 trace 时的相同输入
    onnx_path = "/datasets/model.onnx"
    torch.onnx.export(
        model,
        (dummy_input,),  # 注意是元组
        onnx_path,
        export_params=True,
        opset_version=11,  # 尝试一个较低的 opset 版本
        do_constant_folding=True,
        input_names=['input_dict'],  # 可以自定义
        output_names=['output'],  # 可以自定义
        # dynamic_axes={'input_dict': {0: 'num_points'}, 'output': {0: 'num_points'}} # 如果需要动态轴
    )
    print(f"ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    main_process()