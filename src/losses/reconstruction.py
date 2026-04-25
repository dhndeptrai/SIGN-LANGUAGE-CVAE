"""
Reconstruction Loss (MSE) cho pose prediction.
"""

import torch
import torch.nn as nn


class PoseReconstructionLoss(nn.Module):
    """
    Mean Squared Error loss giữa predicted pose và ground truth.
    
    Hỗ trợ masking để không tính loss trên padding frames.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction (str): 'mean' hoặc 'sum'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pose_pred: torch.Tensor,
        pose_target: torch.Tensor,
        pose_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Tính MSE loss với mask.
        
        Args:
            pose_pred: [batch, seq_len, pose_dim]
            pose_target: [batch, seq_len, pose_dim]
            pose_mask: [batch, seq_len] - Boolean mask (True = valid)
            
        Returns:
            loss: Scalar tensor
        """
        # MSE per element
        mse = (pose_pred - pose_target) ** 2  # [batch, seq_len, pose_dim]
        
        if pose_mask is not None:
            # Apply mask
            mask_expanded = pose_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            mse = mse * mask_expanded
            
            # Reduction
            if self.reduction == 'mean':
                total_elements = mask_expanded.sum()
                loss = mse.sum() / (total_elements + 1e-8)
            else:
                loss = mse.sum()
        else:
            loss = mse.mean() if self.reduction == 'mean' else mse.sum()
        
        return loss