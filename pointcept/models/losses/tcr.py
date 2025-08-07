import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class ClassAwareTCRLoss(nn.Module):
    """Class-Aware Total Coding Rate (TCR) Loss.

    This loss encourages features within each class to span a diverse subspace
    (high coding rate) while preventing features from collapsing. It computes
    the TCR for each semantic class separately based on ground truth labels.
    This implementation handles batched data (e.g., point clouds or image patches).
    """

    def __init__(self,
                 feature_dim=64,
                 num_classes=4,  # Number of classes
                 epsilon=0.2,  # Distortion size for TCR
                 lambda_tcr=0.05,  # Weight for TCR loss
                 min_samples=10,  # Minimum samples per class to compute TCR
                 max_samples_per_class=3000,
                 loss_weight=1.0,
                 ignore_index=-1):  # Index to ignore in target
        """
        Args:
            feature_dim (int): Dimension of the input features (D).
            num_classes (int): Number of classes (C).
            epsilon (float): Distortion size for TCR calculation.
            lambda_tcr (float): Weight for the TCR regularization term.
            min_samples (int): Minimum number of samples required per class to compute TCR.
            loss_weight (float): Overall weight for this loss.
            ignore_index (int): Index to ignore in target (e.g., for void labels).
        """
        super(ClassAwareTCRLoss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.lambda_tcr = lambda_tcr
        self.min_samples = min_samples
        self.max_samples_per_class = max_samples_per_class
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        # Ensure epsilon is a tensor for device handling
        if not isinstance(self.epsilon, torch.Tensor):
            self.epsilon = torch.tensor(float(self.epsilon))

    def forward(self, pred, target, feat, batch):
        """
        Computes the Class-Aware TCR Loss.

        Args:
            feat (torch.Tensor): Point/patch features from the model.
                Shape: (N, D) where N is total number of points/patches, D is feature dim.
            target (torch.Tensor): Ground truth labels for each point/patch.
                Shape: (N,).
            batch (torch.Tensor): Batch index for each point/patch.
                Shape: (N,). E.g., [0, 0, ..., 1, 1, ...].

        Returns:
            torch.Tensor: Computed TCR loss scalar.
        """
        # --- CRITICAL: Ensure tensors are on the same device ---
        if isinstance(self.epsilon, torch.Tensor) and self.epsilon.device != feat.device:
            self.epsilon = self.epsilon.to(feat.device)
        # --- END CRITICAL DEVICE ALIGNMENT ---

        # --- 1. Filter out ignored points ---
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return feat.new_tensor(0.0) * self.loss_weight * self.lambda_tcr

        valid_feat = feat[valid_mask]  # (N_valid, D)
        valid_targets = target[valid_mask]  # (N_valid,)
        valid_batch = batch[valid_mask]  # (N_valid,)

        # Ensure targets are within valid range for indexing
        valid_targets = valid_targets.clamp(0, self.num_classes - 1)

        # --- 2. Get unique batch indices ---
        unique_batches = torch.unique(valid_batch)
        total_tcr_loss = 0.0
        batch_count = 0

        # --- 3. Process each batch separately ---
        for b_idx in unique_batches:
            batch_mask = valid_batch == b_idx
            if not batch_mask.any():
                continue

            batch_feat = valid_feat[batch_mask]  # (M_b, D)
            batch_targets = valid_targets[batch_mask]  # (M_b,)

            # Compute TCR for this batch
            batch_tcr_loss = self._compute_batch_tcr(batch_feat, batch_targets)
            total_tcr_loss += batch_tcr_loss
            batch_count += 1

        # --- 4. Average loss over batches ---
        if batch_count > 0:
            avg_tcr_loss = total_tcr_loss / batch_count
        else:
            avg_tcr_loss = feat.new_tensor(0.0)

        # --- 5. Apply weights ---
        final_loss = self.loss_weight * self.lambda_tcr * avg_tcr_loss

        return final_loss

    def _compute_batch_tcr(self, features, labels):
        """
        Computes the Class-Aware TCR Loss for a single batch.
        """
        total_tcr_loss = 0.0
        valid_classes_count = 0

        for cls in range(self.num_classes):
            cls_mask = labels == cls
            num_cls_samples = cls_mask.sum().item() # 转为 Python int 更安全

            if num_cls_samples >= self.min_samples:
                cls_features = features[cls_mask] # (N_cls, D)

                # --- 新增：子采样逻辑 ---
                if self.max_samples_per_class is not None and num_cls_samples > self.max_samples_per_class:
                    # 随机选择 self.max_samples_per_class 个样本
                    indices = torch.randperm(num_cls_samples, device=cls_features.device)[:self.max_samples_per_class]
                    cls_features = cls_features[indices]
                    # 更新样本数
                    num_cls_samples = self.max_samples_per_class
                # --- 结束：子采样逻辑 ---

                cls_features = F.normalize(cls_features, p=2, dim=1)

                try:
                    tcr_cls = self._compute_tcr(cls_features)

                    min_tcr = 0.5 * torch.log(torch.tensor(float(self.feature_dim), device=cls_features.device))
                    deficit = torch.clamp(min_tcr - tcr_cls, min=0.0)
                    cls_tcr_loss = deficit

                    total_tcr_loss += cls_tcr_loss
                    valid_classes_count += 1

                except Exception as e:

                    # print(f"Warning: Failed to compute TCR for class {cls} in batch: {e}")
                    pass

        if valid_classes_count > 0:
            avg_tcr_loss = total_tcr_loss / valid_classes_count
        else:
            avg_tcr_loss = features.new_tensor(0.0)

        return avg_tcr_loss

    def _compute_tcr(self, Z):
        """
        Computes the Total Coding Rate (TCR) for a set of normalized features.

        Args:
            Z (torch.Tensor): Normalized feature matrix of shape (N, D).

        Returns:
            torch.Tensor: Scalar TCR value.
        """
        N, D = Z.shape

        if N == 0:
            return Z.new_tensor(0.0)

        epsilon_val = self.epsilon.item() if isinstance(self.epsilon, torch.Tensor) else self.epsilon
        if epsilon_val <= 0:
            raise ValueError("epsilon must be a positive value")

        try:
            # 可选：如果 N 远大于 D，考虑使用 Z^T Z 的特征值 (需要数学推导确认等价性)
            # 但对于 EMP-SSL 的典型设置 (N=200 patches, D=512 projector output)，N 不一定远大于 D
            # 因此，子采样是更直接的方法。
            ZZt = torch.mm(Z, Z.t())  # (N, N)
            # cov_term = (D / (N * epsilon_val**2)) * ZZt # 原始公式 (1) 看起来是这样，但论文文本是 d/(b*eps^2)
            # 根据论文公式 (1): R(Z) = 1/2 * log det(I + (d / (b * ε^2)) * ZZ^T)
            # 其中 b 是 batch size (这里对应于 N, 当前类的样本数), d 是特征维度 D
            cov_term = (D / (N * epsilon_val ** 2)) * ZZt

            I = torch.eye(N, device=Z.device, dtype=Z.dtype)  # 注意：这里是 NxN 的单位矩阵
            matrix = I + cov_term  # (N, N)

            # 添加小的对角线项以增强数值稳定性
            # matrix = matrix + 1e-6 * I # 这个 I 是 NxN 的，可能太大。添加到对角线即可
            # 更常见的做法是直接在对角线上加一个小值
            matrix.diagonal().add_(1e-6)  # 等价于 matrix += 1e-6 * torch.eye(N, device=matrix.device)

            # 计算 log determinant
            # 使用 SVD for better numerical stability
            try:
                # SVD of (N, N) matrix
                _, S, _ = torch.svd(matrix)
                # logdet = torch.sum(torch.log(S))
                # 为防止 log(0) 或 log(极小值) 导致的 nan/inf，可以加一个小的正值
                logdet = torch.sum(torch.log(S + 1e-12))  # 添加小值到 S
            except:
                try:
                    # Fallback to torch.logdet, but it's less stable
                    logdet = torch.logdet(matrix)
                except:
                    logdet = Z.new_tensor(0.0)  # Fallback

            tcr = 0.5 * logdet
            return tcr
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM in _compute_tcr for tensor of shape {Z.shape}. Consider reducing max_samples_per_class.")
            raise e  # 重新抛出 OOM 错误以便上层处理或调试

        return Z.new_tensor(0.0)
