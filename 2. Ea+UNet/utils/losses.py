import torch
import torch.nn as nn
import cv2
import numpy as np
from scipy import ndimage

def extract_edge_from_mask(mask, num_classes=4):
    """
    Extract external edges from multiclass segmentation mask.
    Edge width = 1 pixel.
    """
    if mask.ndim == 4:
        mask = mask.squeeze(1)

    batch_size, h, w = mask.shape
    edges = torch.zeros((batch_size, 1, h, w), dtype=torch.float32, device=mask.device)
    mask_np = mask.cpu().numpy().astype(np.uint8)

    for b in range(batch_size):
        for c in range(1, num_classes):
            binary_mask = (mask_np[b] == c).astype(np.uint8) * 255
            if binary_mask.sum() > 0:
                # Apply Canny edge detection
                edge = cv2.Canny(binary_mask, threshold1=100, threshold2=200)
                # Canny outputs 255 for edges, convert to binary 0/1
                edge = (edge > 0).astype(np.float32)
                edges[b, 0] = torch.max(edges[b, 0], torch.from_numpy(edge).to(mask.device))

    return edges


def distance_transform(mask, num_classes=4):
    """
    Compute exactly normalized inner/outer signed distance bounds for boundary loss.
    """
    if mask.ndim == 4:
        mask = mask.squeeze(1)

    batch_size, h, w = mask.shape
    dt_maps = torch.zeros((batch_size, 1, h, w), dtype=torch.float32, device=mask.device)
    mask_np = mask.cpu().numpy()

    for b in range(batch_size):
        foreground = (mask_np[b] > 0)
        if not foreground.any():
            dt_maps[b, 0] = 1.0  # Entirely background
            continue

        inner_dt = ndimage.distance_transform_edt(foreground)
        outer_dt = ndimage.distance_transform_edt(~foreground)

        inner_max = inner_dt.max() if inner_dt.max() > 0 else 1.0
        outer_max = outer_dt.max() if outer_dt.max() > 0 else 1.0

        inner_dt_norm = inner_dt / inner_max
        outer_dt_norm = outer_dt / outer_max

        sdf = outer_dt_norm - inner_dt_norm
        dt_maps[b, 0] = torch.from_numpy(sdf).to(mask.device)

    return dt_maps


class BoundaryLoss(nn.Module):
    """
    Combined Edge-Aware Loss replacing obsolete EdgeBCEDiceLoss.
    Optimizes lateral_edge using true boundaries generated from masks.
    Avoids zero-collapse by combining explicit boundary matching (BCE) with 
    distance map regularization (MSE/SDF).
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, edge_pred, target_mask):
        """
        edge_pred: [B, 1, H, W]  Logits from edge branch
        target_mask: [B, H, W] Ground truth mask
        """
        # 1. Compute exact 1-pixel boundary mask from ground truth
        true_edges = extract_edge_from_mask(target_mask, num_classes=self.num_classes)

        # 2. Compute signed distance transform (SDF) from ground truth
        dt_map = distance_transform(target_mask, num_classes=self.num_classes)

        # 3. BCE Loss: force edge_pred logit to precisely match literal 1-pixel boundary
        loss_bce = self.bce(edge_pred, true_edges)

        # 4. SDF Loss: force predicted edge probabilities (sigmoid) to be higher near boundaries
        #    and heavily penalized further away (dt_map is large).
        edge_prob = torch.sigmoid(edge_pred)
        loss_dt = torch.mean((edge_prob * dt_map) ** 2)

        return loss_bce + loss_dt


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Multiclass Dice Loss.
    Weights each class inversely proportional to its volume.
    Crucial for handling extreme class imbalances (e.g. tiny Enhancing Tumors).
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1e-5):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate class weights (inverse of volume squared)
        volumes = targets_one_hot.sum(dim=(2, 3)) # [B, C]
        weights = 1.0 / (volumes ** 2 + smooth)
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        cardinality = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        # Weight the intersection and cardinality
        weighted_intersection = (weights * intersection).sum(dim=1)
        weighted_cardinality = (weights * cardinality).sum(dim=1)
        
        dice = (2. * weighted_intersection + smooth) / (weighted_cardinality + smooth)
        
        loss = 1 - dice
        return loss.mean()
