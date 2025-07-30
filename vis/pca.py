import torch
import numpy as np
from pointcept.models.utils.structure import Point

try:
    import flash_attn
except ImportError:
    flash_attn = None

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
        "coord": torch.from_numpy(coord).float(),
        "color": torch.from_numpy(color).float(),
        "segment": torch.from_numpy(segment).long()
    }
    






if __name__ == "__main__":
    data_root = r"D:\04-Datasets\navarra-test\processed\test\02"

