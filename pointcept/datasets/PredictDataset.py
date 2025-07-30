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



import pdal

try:
    import pointops
except:
    pointops = None
from .defaults import DefaultDataset
from .builder import DATASETS



@DATASETS.register_module()
class PredictDataset(DefaultDataset):
    def __init__(self,
                 split="test",
                 data_root="",
                 transforms=None,
                 test_mode=True,
                 test_cfg=None,
                 cache=False,
                 ignore_index=-1,
                 loop=1,):
        super(PredictDataset, self).__init__(split, data_root, transforms, test_mode,
                                             test_cfg, cache, ignore_index, loop)



        self.center = torch.zeros((1, 3))

        # 原始点云，用 pdal 存储
        self.srs: str = ''
        self.color: np.ndarray = None
        self.pos: np.ndarray = None
        self.segment: np.ndarray = None
        self.prepare_data()

    def prepare_data(self):
        print("Preparing data...")
        extension_name = os.path.splitext(self.test_cfg.test_file)[1].lower()
        assert extension_name == '.las' or extension_name == '.laz'
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader.las(filename=self.test_cfg.test_file, nosrs=True)
        pipeline |= pdal.Filter.stats(dimensions="Intensity,Red,Blue,Green")
        pipeline |= pdal.Filter.voxelcenternearestneighbor(cell=self.cfg.grid_size)
        pipeline |= pdal.Filter.outlier(method="statistical", multiplier=3.0)
        pipeline |= pdal.Filter.range(limits="Classification![7:7]")
        pipeline |= pdal.Filter.assign(value=[
            f"Classification = 0 WHERE Classification == 2",
            f"Classification = 1 WHERE ((Classification >= 3) && (Classification <= 5))",
            f"Classification = 2 WHERE Classification == 6",
            f"Classification = 3 WHERE Classification == 8"
        ])
        pipeline.execute()
        metadata = pipeline.metadata['metadata']
        bounds = metadata['readers.las']
        minx = bounds['minx']
        miny = bounds['miny']
        minz = bounds['minz']
        self.center = torch.tensor([minx, miny, minz], dtype=torch.float64)
        self.srs = metadata['readers.las']['comp_spatialreference']

        arrays = pipeline.arrays[0]

        stats = metadata['filters.stats']['statistic'][1]
        r = self.normalize_attributes(arrays['Red'], 0.0, (float)(stats['maximum']), stats['average'],
                                 stats['stddev'])
        stats = metadata['filters.stats']['statistic'][2]
        g = self.normalize_attributes(arrays['Green'], 0.0, (float)(stats['maximum']), stats['average'],
                                 stats['stddev'])
        stats = metadata['filters.stats']['statistic'][3]
        b = self.normalize_attributes(arrays['Blue'], 0.0, (float)(stats['maximum']), stats['average'],
                                 stats['stddev'])

        self.color = np.concatenate([r, g, b], axis=-1)

        self.pos = np.concatenate([
            np.expand_dims(arrays['X'] - minx, 1),
            np.expand_dims(arrays['Y'] - miny, 1),
            np.expand_dims(arrays['Z'] - minz, 1)
        ], axis=-1).astype(np.float32)

        self.segment = arrays['Classification'].reshape(-1, 1)


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

    def get_data(self,idx):
        data_dict = {}
        data_dict["coord"] = self.pos.astype(np.float32)
        data_dict["color"] = self.color.astype(np.float32)
        data_dict["segment"] = self.segment.reshape([-1]).astype(np.int32)
        return data_dict

