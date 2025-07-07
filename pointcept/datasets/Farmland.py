import os
from .defaults import DefaultDataset
from .builder import DATASETS

import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset
from .transform import Compose, TRANSFORMS

@DATASETS.register_module()
class FarmlandDataset(Dataset):
    lock = None
    def __init__(self,num_points_per_step, transform=None,test_mode=False, split='train', sample_method = 'ball_sample'):
        super().__init__()
        self.inputs = []
        self.segments = []
        self.kdtrees = []
        self.colors = []
        self.potentials = []
        self.min_potentials = []
        self.class_weights = []
        self.load_dataset()
        # TODO potentials
        self.min_potentials=[]
        self.argmin_potentials=[]
        self.transform = Compose(transform)

        # todo
        self.num_points_per_step = num_points_per_step
        self.sample_method = sample_method
        self.test_mode = test_mode
        self.split = split

    # def get_data_name(self, idx):
    #     remain, room_name = os.path.split(self.data_list[idx % len(self.data_list)])
    #     remain, area_name = os.path.split(remain)
    #     return f"{area_name}-{room_name}"


    def load_dataset(self):
        sub_dir = self.split #
        # sub_dir下面所有的文件夹名字，不迭代
        for dir in os.listdir(sub_dir):
            if os.path.isdir(os.path.join(sub_dir, dir)):
                coord_path = os.path.join(sub_dir, dir,'coord.npy')
                color_path = os.path.join(sub_dir, dir,'color.npy')
                segment_path = os.path.join(sub_dir, dir,'segment.npy')
                kdtree_path = os.path.join(sub_dir, dir,'kdtree.pkl')

                with open(kdtree_path, 'rb') as f:
                    kdtree = pickle.load(f)
                coord = np.load(coord_path).astype(np.float32)
                color = np.load(color_path).astype(np.float32)
                segment = np.load(segment_path).astype(np.int16)

                self.kdtrees.append(kdtree)
                self.inputs.append(coord)
                self.segments.append(segment)
                self.colors.append(color)

    @staticmethod
    def shuffle_idx(x):
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    def get_regular_neighbor(self):
        assert self.lock is not None
        while True:
            with self.lock:
                cloud_idx = int(torch.argmin(self.min_potentials))
                point_idx = int(self.argmin_potentials[cloud_idx])
                cloud_data = self.inputs[cloud_idx]
                color = self.colors[cloud_idx]
                segment = self.segments[cloud_idx]
                kdtree_data: KDTree = self.kdtrees[cloud_idx]
                # 获取中心点
                center = cloud_data[point_idx].reshapr(1, -1)
                center += np.random.rand(1, 3) * 0.5
                num_knn = self.num_points_per_step
                num_knn = min(num_knn, cloud_data.shape[0])

                dists, indices = kdtree_data.query(center, k=num_knn)

                indices = indices[0]
                dists = dists[0]

                if indices.shape[0] > self.num_points_per_step:
                    indices = indices[:self.num_points_per_step]
                    dists = dists[:self.num_points_per_step]

                dists = np.square(dists)
                max_dist = dists[-1] + 0.01

                # 进行potential  选点

                # 点越少，权重越小，下次继续选择该点的概率就大
                points_weights = self.segments[indices]

                points_weights = self.class_weights.index_select(-1, points_weights)
                points_weights = points_weights.numpy()

                # 更新 potentials 的权重
                tukeys = np.square(1 - dists / max_dist) * points_weights

                self.potentials[cloud_idx][indices] += tukeys
                min_point_idx = torch.argmin(self.potentials[cloud_idx])
                self.min_potentials[[cloud_idx]] = self.potentials[cloud_idx][min_point_idx]
                self.argmin_potentials[[cloud_idx]] = min_point_idx

                if indices.shape[0] > 100:
                    # 点数太少重新选择点
                    break

        indices = self.shuffle_idx(indices)

        # 输出一直要为numpy
        coord = cloud_data[indices, :] - center
        color = color[indices, :]
        segment = segment[indices, :]
        data_dict = {}
        data_dict['coord'] = coord.astype(np.float32)
        data_dict['segment'] = segment.astype(np.float32)
        data_dict['color'] = color.astype(np.float32)

        return data_dict

    def get_data(self, cloud_idx):
        data_dict = {}
        if self.sample_method == 'ball_sample':
            data_dict = self.get_regular_neighbor()
        return data_dict

    def prepare_train_data(self, cloud_idx):
        data_dict = self.get_data(cloud_idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, cloud_idx):
        data_dict = self.get_data(cloud_idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)










