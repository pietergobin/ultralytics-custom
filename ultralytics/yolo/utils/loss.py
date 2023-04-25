# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


class EdgeCoordinateLoss(nn.Module):

    def __init__(self, edge_threshold: int) -> None:
        super().__init__()
        self.threshold = edge_threshold

    def forward(self, pred_bboxes, target_bboxes):
        """
        Calculates the sum of euclidian distances between the target cornerpoint and the predicted cornerpoint
        """
        bb_x_topleft = target_bboxes[..., 0]
        bb_y_topleft = target_bboxes[..., 1]

        quadrant_0_mask = (bb_y_topleft < self.threshold) & (bb_x_topleft < self.threshold)
        quadrant_1_mask = (bb_y_topleft < self.threshold) & (bb_x_topleft > self.threshold)
        quadrant_2_mask = (bb_y_topleft > self.threshold) & (bb_x_topleft < self.threshold)
        quadrant_3_mask = (bb_y_topleft > self.threshold) & (bb_x_topleft > self.threshold)

        quadrant_0_pred, quadrant_0_target = pred_bboxes[quadrant_0_mask], target_bboxes[quadrant_0_mask]
        quadrant_1_pred, quadrant_1_target = pred_bboxes[quadrant_1_mask], target_bboxes[quadrant_1_mask]
        quadrant_2_pred, quadrant_2_target = pred_bboxes[quadrant_2_mask], target_bboxes[quadrant_2_mask]
        quadrant_3_pred, quadrant_3_target = pred_bboxes[quadrant_3_mask], target_bboxes[quadrant_3_mask]

        delta_x = self.get_euclidean_dist(quadrant_0_pred[..., 2], quadrant_0_target[..., 2]).add(
            self.get_euclidean_dist(quadrant_1_pred[..., 0], quadrant_1_target[..., 0])).add(
            self.get_euclidean_dist(quadrant_2_pred[..., 2], quadrant_2_target[..., 2])).add(
            self.get_euclidean_dist(quadrant_3_pred[..., 0], quadrant_3_target[..., 0]))

        delta_y = self.get_euclidean_dist(quadrant_0_pred[..., 3], quadrant_0_target[..., 3]).add(
                  self.get_euclidean_dist(quadrant_1_pred[..., 3], quadrant_1_target[..., 3])).add(
                  self.get_euclidean_dist(quadrant_2_pred[..., 1], quadrant_2_target[..., 1])).add(
                  self.get_euclidean_dist(quadrant_3_pred[..., 1], quadrant_3_target[..., 1]))

        return torch.sum(delta_x), torch.sum(delta_y)

    @staticmethod
    def get_euclidean_dist(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return (t2 - t1).pow(2).sqrt()

    @staticmethod
    def get_xy_coordinates(BB_truth, BB_pred, quadrant):
        if quadrant == 0:
            return BB_truth[2], BB_truth[3], BB_pred[2], BB_pred[3]
        elif quadrant == 1:
            return BB_truth[0], BB_truth[3], BB_pred[0], BB_pred[3]
        elif quadrant == 2:
            return BB_truth[2], BB_truth[1], BB_pred[2], BB_pred[1]
        elif quadrant == 3:
            return BB_truth[0], BB_truth[1], BB_pred[0], BB_pred[1]
        else:
            return None, None, None, None
