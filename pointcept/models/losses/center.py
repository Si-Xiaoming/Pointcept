import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

@LOSSES.register_module()
class DynamicCenterLoss(nn.Module):
    """Dynamic Center Loss for Point Cloud Semantic Segmentation.
    ... (docstring remains the same) ...
    """
    """Dynamic Center Loss for Point Cloud Semantic Segmentation.

    This loss encourages points of the same class to be close to their dynamic
    class center (intra-class compactness) and pushes centers of different
    classes apart (inter-class separation).

    The class centers can be either learnable parameters or updated via
    Exponential Moving Average (EMA) based on the features of the current batch.
    This implementation handles batched point cloud data.
    """

    def __init__(self,
                 feature_dim=64,
                 num_classes=4,
                 center_type='ema', # 'learnable' or 'ema'
                 alpha=0.99,        # EMA momentum for center update
                 margin=0.50,        # Margin for inter-class separation
                 intra_weight=1.0,  # Weight for intra-class loss
                 inter_weight=1.0,  # Weight for inter-class loss
                 loss_weight=0.01,
                 ignore_index=-1):
        """
        Args:
            feature_dim (int): Dimension of the input point features (D).
            num_classes (int): Number of classes (C).
            center_type (str): How to maintain centers.
                               'learnable': Centers are learnable parameters.
                               'ema': Centers are updated via EMA.
            alpha (float): Momentum for EMA center update (used if center_type='ema').
            margin (float): Margin for the inter-class separation term.
            intra_weight (float): Weight for the intra-class compactness loss.
            inter_weight (float): Weight for the inter-class separation loss.
            loss_weight (float): Overall weight for this loss.
            ignore_index (int): Index to ignore in target (e.g., for padding).
        """
        super(DynamicCenterLoss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.center_type = center_type
        self.alpha = alpha
        self.margin = margin
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        if self.center_type == 'learnable':
            # Centers are learnable parameters
            self.register_parameter(
                'centers',
                nn.Parameter(torch.randn(self.num_classes, self.feature_dim))
            )
        elif self.center_type == 'ema':
            # Centers are buffers, updated via EMA
            self.register_buffer(
                'centers',
                torch.randn(self.num_classes, self.feature_dim)
            )
            # Optional: Keep track of counts for better EMA (not strictly necessary)
            # self.register_buffer('center_counts', torch.zeros(self.num_classes))
        else:
            raise ValueError(f"center_type must be 'learnable' or 'ema', got {center_type}")

        # Ensure alpha is on the same device as centers
        if not isinstance(self.alpha, torch.Tensor):
             self.alpha = torch.tensor(self.alpha)


    def forward(self, pred, target, feat, batch):
        """
        Computes the Dynamic Center Loss for batched point cloud data.

        Args:
            feat (torch.Tensor): Point features from the model.
                Shape: (N, D) where N is total number of points, D is feature dim.
            target (torch.Tensor): Ground truth labels for each point.
                Shape: (N,).
            batch (torch.Tensor): Batch index for each point.
                Shape: (N,). E.g., [0, 0, ..., 1, 1, ...].

        Returns:
            torch.Tensor: Computed loss scalar.
        """
        # --- CRITICAL: Ensure tensors are on the same device ---
        # Move centers and alpha to the same device as input features (feat)
        if self.centers.device != feat.device:
            self.centers = self.centers.to(feat.device)
        if isinstance(self.alpha, torch.Tensor) and self.alpha.device != feat.device:
            self.alpha = self.alpha.to(feat.device)
        # --- END CRITICAL DEVICE ALIGNMENT ---

        # Now proceed with the rest of the forward pass logic
        if self.center_type == 'learnable':
            centers = self.centers
        elif self.center_type == 'ema':
            centers = self.centers

        # --- 1. Filter out ignored points ---
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return feat.new_tensor(0.0) # This correctly uses the device of feat

        valid_feat = feat[valid_mask]
        valid_targets = target[valid_mask]
        valid_batch = batch[valid_mask]

        # --- 2. Get unique batch indices ---
        unique_batches = torch.unique(valid_batch)
        total_intra_loss = 0.0
        total_inter_loss = 0.0
        batch_count = 0

        # --- 3. Process each batch separately ---
        for b_idx in unique_batches:
            batch_mask = valid_batch == b_idx
            if not batch_mask.any():
                continue

            batch_feat = valid_feat[batch_mask]
            batch_targets = valid_targets[batch_mask] # This is now guaranteed to be on the same device as feat

            # --- 3a. Intra-class Compactness Loss ---
            # Now this line should work because both centers and batch_targets are on the same device (feat.device)
            batch_centers = centers[batch_targets] # (M_b, D) - Should work now

            intra_loss_batch = F.mse_loss(batch_feat, batch_centers, reduction='none').sum(dim=1).mean()
            total_intra_loss += intra_loss_batch

            # --- 3b. Inter-class Separation Loss (logic remains the same) ---
            # ... (rest of inter-class loss calculation) ...
            unique_labels_in_batch = torch.unique(batch_targets)
            if len(unique_labels_in_batch) > 1:
                 # For each class in the batch, compute its center
                 batch_class_centers = []
                 batch_class_indices = []
                 for c in unique_labels_in_batch:
                     class_mask = batch_targets == c
                     if class_mask.sum() > 0:
                         class_center = batch_feat[class_mask].mean(dim=0) # (D,)
                         batch_class_centers.append(class_center)
                         batch_class_indices.append(c)

                 if len(batch_class_centers) > 1:
                     batch_class_centers = torch.stack(batch_class_centers) # (num_unique_classes_in_batch, D)
                     # batch_class_indices = torch.tensor(batch_class_indices, device=feat.device) # Not strictly needed here

                     diff = batch_class_centers.unsqueeze(1) - batch_class_centers.unsqueeze(0)
                     distances = torch.norm(diff, p=2, dim=2)

                     num_classes_batch = batch_class_centers.shape[0]
                     off_diag_mask = ~torch.eye(num_classes_batch, dtype=torch.bool, device=feat.device)

                     margin_diff = self.margin - distances
                     inter_loss_per_pair = torch.clamp(margin_diff[off_diag_mask], min=0.0)
                     inter_loss_batch = inter_loss_per_pair.mean() if inter_loss_per_pair.numel() > 0 else feat.new_tensor(0.0)
                     total_inter_loss += inter_loss_batch

            batch_count += 1

        # --- 4. Average loss over batches ---
        if batch_count > 0:
            avg_intra_loss = total_intra_loss / batch_count
            avg_inter_loss = total_inter_loss / batch_count
        else:
            avg_intra_loss = feat.new_tensor(0.0)
            avg_inter_loss = feat.new_tensor(0.0)

        # --- 5. Total Loss ---
        total_loss = self.intra_weight * avg_intra_loss + self.inter_weight * avg_inter_loss

        # --- 6. Update centers using EMA if applicable ---
        if self.center_type == 'ema' and self.training:
            self._update_centers_ema(valid_feat, valid_targets)

        return self.loss_weight * total_loss

    def _update_centers_ema(self, features, labels):
        """
        Updates the global class centers using Exponential Moving Average (EMA).
        """
        # Ensure operation is detached from the main computation graph
        with torch.no_grad():
            # Ensure centers are on the same device as features
            # (This check might be redundant if forward() already moved them, but good for safety)
            if self.centers.device != features.device:
                 self.centers = self.centers.to(features.device)
                 if isinstance(self.alpha, torch.Tensor):
                      self.alpha = self.alpha.to(features.device)

            for c in range(self.num_classes):
                class_mask = (labels == c)
                if class_mask.sum() > 0:
                    new_center = features[class_mask].mean(dim=0) # (D)
                    # Ensure alpha is used correctly (scalar tensor or float)
                    alpha_val = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
                    self.centers[c] = alpha_val * self.centers[c] + (1 - alpha_val) * new_center
                    # Optional normalization
                    # self.centers[c] = F.normalize(self.centers[c], dim=0, p=2)

    # ... (rest of the class, like extra_repr, remains the same) ...
