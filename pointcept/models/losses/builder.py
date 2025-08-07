"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry
from pointcept.models import losses
LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred=None, target=None, feat=None, batch=None):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            if isinstance(c, losses.DynamicCenterLoss) or isinstance(c, losses.ClassAwareTCRLoss):
                loss += c(pred, target, feat, batch)

            else:
                loss += c(pred, target)
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
