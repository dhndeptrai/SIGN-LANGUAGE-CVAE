"""
Pose Decoder: Giải mã latent z và context c thành chuỗi pose.

Sử dụng GRU autoregressive với teacher forcing trong training.
"""

import torch
import torch.nn as nn


class PoseDecoder(nn.Module):
    """
    Decoder sinh chuỗi pose từ latent z và context c.
    
    Architecture:
        [z; c] -> FC -> GRU (autoregressive) -> FC -> Pose sequence
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        context_dim: int = 512,
        pose_dim: int = 225,
        hidden_dim: int = 512,
        num_layers: int = 2,
        max_len: int = 300
    ):
        """
        Args:
            latent_dim (int): Dimension của latent z
            context_dim (int): Dimension của context c
            pose_dim (int): Dimension của output pose
            hidden_dim (int): Hidden dimension của GRU
            num_layers (int): Số lớp GRU
            max_len (int): Độ dài tối đa sequence sinh ra
        """
        super().__init__()
        
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # FC để kết hợp z và c thành initial hidden state
        self.fc_init = nn.Linear(latent_dim + context_dim, hidden_dim * num_layers)
        
        # GRU decoder
        self.gru = nn.GRU(
            input_size=pose_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, pose_dim)
    
    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        target_pose: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Decode latent z thành pose sequence.
        
        Args:
            z: Latent vector [batch, latent_dim]
            context: Context vector [batch, context_dim]
            target_pose: Ground truth pose [batch, seq_len, pose_dim] (for teacher forcing)
            teacher_forcing_ratio: Xác suất sử dụng ground truth frame thay vì prediction
            
        Returns:
            pose_pred: Predicted pose [batch, seq_len, pose_dim]
        """
        batch_size = z.size(0)
        seq_len = target_pose.size(1) if target_pose is not None else self.max_len
        
        # Kết hợp z và c để tạo initial hidden state
        combined = torch.cat([z, context], dim=-1)  # [batch, latent_dim + context_dim]
        h0 = self.fc_init(combined)  # [batch, hidden_dim * num_layers]
        h0 = h0.view(batch_size, -1, self.hidden_dim).transpose(0, 1).contiguous()  # [num_layers, batch, hidden_dim]
        
        # Initial input: zero pose
        decoder_input = torch.zeros(batch_size, 1, self.pose_dim, device=z.device)  # [batch, 1, pose_dim]
        
        hidden = h0
        outputs = []
        
        for t in range(seq_len):
            # GRU step
            gru_out, hidden = self.gru(decoder_input, hidden)  # gru_out: [batch, 1, hidden_dim]
            
            # Predict pose
            pose_pred = self.fc_out(gru_out.squeeze(1))  # [batch, pose_dim]
            outputs.append(pose_pred.unsqueeze(1))
            
            # Teacher forcing decision
            if target_pose is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_pose[:, t:t+1, :]  # Use ground truth
            else:
                decoder_input = pose_pred.unsqueeze(1)  # Use prediction
        
        pose_pred = torch.cat(outputs, dim=1)  # [batch, seq_len, pose_dim]
        return pose_pred