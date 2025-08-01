import os
from .defaults import DefaultDataset
from .builder import DATASETS

@DATASETS.register_module()
class NavarraDataset(DefaultDataset):

    def __init__(self, split='train',
                 data_root="datasets/navarra",
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 cache=False,
                 ignore_index=-1,
                 loop=1):
        data_root = os.path.join(data_root, "processed")
        super(NavarraDataset, self).__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            cache=cache,
            ignore_index=ignore_index,
            loop=loop
        )

    def get_data_name(self, idx):
        data_name = os.path.split(
            self.data_list[idx % len(self.data_list)]
        )
        return f"{data_name[1]}"