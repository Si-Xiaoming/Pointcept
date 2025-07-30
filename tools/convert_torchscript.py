import argparse
import os

import torch.jit

from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
import pointcept.utils.comm as comm
from collections import OrderedDict

def load_model():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    cfg = default_setup(cfg)

    model = build_model(cfg.model)

    print("loaded init model")
    if os.path.isfile(cfg.weight):

        checkpoint = torch.load(cfg.weight, weights_only=False)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                if comm.get_world_size() > 1:
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
            weight[key] = value
        model.load_state_dict(weight, strict=True)

    else:
        raise RuntimeError("=> No checkpoint found.")
    return model

def convert_torchscript(output):
    sample_data = {"coord": torch.randn(2, 3),
                   "color": torch.randn(2, 3),
                   "batch": torch.Tensor([0., 1.])}
    model = load_model()
    model.eval()

    script_model = torch.jit.trace(model, sample_data)

    torch.jit.save(script_model, output)



if __name__ == '__main__':
    output = r"datasets/out.pt"
    convert_torchscript(output)

    