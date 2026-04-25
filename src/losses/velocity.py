"""
Velocity Smoothness Loss để giảm hiệu ứng giật (jitter).

Penalize sự thay đổi đột ngột giữa các frame liên tiếp.
Tham khảo: Baltatzis et al. (2024) - Neural Sign Actors
"""

import torch
import torch.nn as nn


class VelocityLoss(nn.Module):
    """
    Smoothness loss dựa trên vận tốc (đạo hàm bậc 1).
    
    Công thức: L_vel = ||pose[t] - pose[t-1]||^2
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
        pose_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Tính velocity loss.
        
        Args:
            pose_pred: [batch, seq_len, pose_dim]
            pose_mask: [batch, seq_len] - Boolean mask
            
        Returns:
            vel_loss: Scalar tensor
        """
        # Tính velocity (difference giữa frame t và t-1)
        velocity = pose_pred[:, 1:, :] - pose_pred[:, :-1, :]  # [batch, seq_len-1, pose_dim]
        
        # L2 norm
        vel_loss = (velocity ** 2).sum(dim=-1)  # [batch, seq_len-1]
        
        if pose_mask is not None:
            # Mask cho velocity (cả 2 frame phải valid)
            vel_mask = pose_mask[:, 1:] & pose_mask[:, :-1]  # [batch, seq_len-1]
            vel_loss = vel_loss * vel_mask.float()
            
            if self.reduction == 'mean':
                total_valid = vel_mask.sum()
                vel_loss = vel_loss.sum() / (total_valid + 1e-8)
            else:
                vel_loss = vel_loss.sum()
        else:
            vel_loss = vel_loss.mean() if self.reduction == 'mean' else vel_loss.sum()
        
        return vel_loss