"""
Velocity và Acceleration Smoothness Loss để giảm hiệu ứng giật (jitter).

Phiên bản nâng cấp so với base:
    - Thêm Acceleration Loss (đạo hàm bậc 2) bên cạnh Velocity Loss (bậc 1).
    - Acceleration Loss penalize sự thay đổi đột ngột của vận tốc giữa
      các frame liên tiếp, tạo chuyển động mượt mà hơn.
    - Thêm Per-Part weighting: tay được weighted cao hơn body vì tay nhỏ hơn
      và dễ bị "dính" hơn khi model không được ràng buộc đủ.

Tham khảo:
    - Baltatzis et al. (2024): Neural Sign Actors - motion smoothness.
    - Harvey et al. (2020): Robust Motion In-Betweening (acceleration constraint).
      https://arxiv.org/abs/2102.04942
"""

import torch
import torch.nn as nn


class VelocityLoss(nn.Module):
    """
    Smoothness loss kết hợp Velocity (bậc 1) và Acceleration (bậc 2).

    Công thức đầy đủ:
        L_vel  = mean || pose[t] - pose[t-1] ||^2
        L_accel= mean || (pose[t+1] - pose[t]) - (pose[t] - pose[t-1]) ||^2
        L_total = L_vel + alpha * L_accel

    Việc thêm L_accel đảm bảo không chỉ biên độ bước nhảy nhỏ (velocity)
    mà còn sự thay đổi của bước nhảy cũng mượt mà (acceleration).

    Ngoài ra, tọa độ được split thành 3 phần (body, left_hand, right_hand)
    và hand được weighted cao hơn body để ép model học cấu trúc ngón tay
    chi tiết hơn.

    Attributes:
        alpha (float): Trọng số của acceleration loss so với velocity loss.
        hand_weight (float): Hệ số nhân thêm cho phần tay so với body.
        reduction (str): 'mean' hoặc 'sum'.

    Tham khảo:
        Baltatzis et al. (2024); Harvey et al. (2020).
    """

    # MediaPipe Holistic layout trong vector 225-dim:
    # Body:       dims 0   .. 98  (33 landmarks * 3)
    # Left hand:  dims 99  .. 161 (21 landmarks * 3)
    # Right hand: dims 162 .. 224 (21 landmarks * 3)
    BODY_DIMS = slice(0, 99)
    LEFT_HAND_DIMS = slice(99, 162)
    RIGHT_HAND_DIMS = slice(162, 225)

    def __init__(
        self,
        alpha: float = 0.5,
        hand_weight: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha (float): Trọng số của acceleration loss.
                           0 = chỉ velocity loss (giống phiên bản cũ).
                           0.5 = cân bằng, khuyến nghị cho sign language.
            hand_weight (float): Hệ số nhân loss cho tay so với body.
                                 2.0 nghĩa là tay được penalize nặng gấp đôi.
            reduction (str): Cách tổng hợp: 'mean' hoặc 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.hand_weight = hand_weight
        self.reduction = reduction

    def _compute_velocity(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Tính velocity: hiệu giữa frame liên tiếp.

        Args:
            pose (torch.Tensor): Shape [batch, seq_len, pose_dim].

        Returns:
            torch.Tensor: Shape [batch, seq_len-1, pose_dim]. Velocity sequence.
        """
        return pose[:, 1:, :] - pose[:, :-1, :]

    def _compute_acceleration(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Tính acceleration: hiệu của velocity liên tiếp (đạo hàm bậc 2).

        Args:
            velocity (torch.Tensor): Shape [batch, seq_len-1, pose_dim].

        Returns:
            torch.Tensor: Shape [batch, seq_len-2, pose_dim]. Acceleration sequence.
        """
        return velocity[:, 1:, :] - velocity[:, :-1, :]

    def _masked_mse(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Tính MSE có mask trên chiều seq và pose_dim.

        Args:
            tensor (torch.Tensor): Shape [batch, T, pose_dim].
            mask (torch.Tensor): Shape [batch, T]. Boolean, True=valid.

        Returns:
            torch.Tensor: Scalar loss.
        """
        squared = (tensor ** 2).sum(dim=-1)  # [batch, T]

        if mask is not None:
            squared = squared * mask.float()
            if self.reduction == 'mean':
                return squared.sum() / (mask.sum().clamp(min=1))
            return squared.sum()

        return squared.mean() if self.reduction == 'mean' else squared.sum()

    def forward(
        self,
        pose_pred: torch.Tensor,
        pose_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Tính tổng smoothness loss = velocity loss + alpha * acceleration loss,
        với per-part weighting cho tay.

        Args:
            pose_pred (torch.Tensor): Shape [batch, seq_len, pose_dim=225].
                                      Chuỗi pose đã được predict bởi decoder.
            pose_mask (torch.Tensor): Shape [batch, seq_len]. Boolean mask.
                                      True = frame hợp lệ, False = padding.

        Returns:
            torch.Tensor: Scalar smoothness loss.
        """
        # ---- Tính velocity và acceleration ----
        velocity = self._compute_velocity(pose_pred)         # [B, T-1, 225]
        acceleration = self._compute_acceleration(velocity)  # [B, T-2, 225]

        # ---- Tạo mask tương ứng ----
        vel_mask = None
        accel_mask = None
        if pose_mask is not None:
            vel_mask = pose_mask[:, 1:] & pose_mask[:, :-1]           # [B, T-1]
            accel_mask = vel_mask[:, 1:] & vel_mask[:, :-1]           # [B, T-2]

        # ---- Per-part velocity loss ----
        # Body
        vel_body = velocity[..., self.BODY_DIMS]
        accel_body = acceleration[..., self.BODY_DIMS]

        # Left hand
        vel_lhand = velocity[..., self.LEFT_HAND_DIMS]
        accel_lhand = acceleration[..., self.LEFT_HAND_DIMS]

        # Right hand
        vel_rhand = velocity[..., self.RIGHT_HAND_DIMS]
        accel_rhand = acceleration[..., self.RIGHT_HAND_DIMS]

        # Velocity loss: body + hand * hand_weight
        vel_loss = (
            self._masked_mse(vel_body, vel_mask)
            + self.hand_weight * self._masked_mse(vel_lhand, vel_mask)
            + self.hand_weight * self._masked_mse(vel_rhand, vel_mask)
        )

        # Acceleration loss: tương tự
        accel_loss = (
            self._masked_mse(accel_body, accel_mask)
            + self.hand_weight * self._masked_mse(accel_lhand, accel_mask)
            + self.hand_weight * self._masked_mse(accel_rhand, accel_mask)
        )

        total_loss = vel_loss + self.alpha * accel_loss
        return total_loss
