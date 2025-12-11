# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn.functional as F


# class RobustCrossEntropyLoss(torch.nn.Module):
#     def __init__(self, label_smoothing: float = 0.0) -> None:
#         super().__init__()
#         self.label_smoothing = label_smoothing

#     def forward(self, lb, reduction="mean"):
#         # lb is lower bound of (`f(x)_y - f(x)_t`)
#         zero_lb = torch.zeros((lb.size(0), 1)).type_as(lb)
#         lb_padded = torch.cat((zero_lb, lb), dim=1)
#         label_zero = torch.zeros((lb.size(0),)).type_as(lb).long()
#         loss = F.cross_entropy(
#             -lb_padded,
#             label_zero,
#             reduction=reduction,
#             label_smoothing=self.label_smoothing,
#         )
#         return loss

class RobustCrossEntropyLoss(torch.nn.Module):
    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, lb, y_smoothed=None, reduction="mean"):
        """
        Args:
            lb: lower bounds of (f(x)_y - f(x)_t), shape [batch_size, num_classes-1]
            y_smoothed: optional smoothed labels, shape [batch_size, num_classes]
            reduction: 'mean' or 'sum'
        """
        # lb is lower bound of (`f(x)_y - f(x)_t`)
        zero_lb = torch.zeros((lb.size(0), 1)).type_as(lb)
        lb_padded = torch.cat((zero_lb, lb), dim=1)  # [batch_size, num_classes]
        
        if y_smoothed is not None:
            # Use soft labels - compute cross entropy manually
            # CE = -sum(p * log(q)) where p=y_smoothed, q=softmax(-lb_padded)
            log_probs = F.log_softmax(-lb_padded, dim=1)
            loss = -(y_smoothed * log_probs).sum(dim=1)  # [batch_size]
            
            if reduction == "mean":
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()
        else:
            # Original behavior with hard labels
            label_zero = torch.zeros((lb.size(0),)).type_as(lb).long()
            loss = F.cross_entropy(
                -lb_padded,
                label_zero,
                reduction=reduction,
                label_smoothing=self.label_smoothing,
            )
        
        return loss
