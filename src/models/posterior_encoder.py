"""
Posterior Encoder: Encode pose sequence và context thành phân phối q(z|x,c).

Chỉ được sử dụng trong quá trình training để học không gian tiềm ẩn.
"""

import torch
import torch.nn as nn


class PosteriorEncoder(nn.Module):
    """
    Posterior Encoder ước lượng q(z|x,c) từ pose thực và context.
    
    Architecture:
        Pose Sequence -> GRU -> Concatenate with Context -> MLP -> (mu, logvar)
    """
    
    def __init__(
        self,
        pose_dim: int = 225,  # 33*3 + 21*3 + 21*3 = 225
        context_dim: int = 512,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2
    ):
        """
        Args:
            pose_dim (int): Dimension của mỗi pose frame (225 cho MediaPipe)
            context_dim (int): Dimension của context vector
            hidden_dim (int): Hidden dimension của GRU
            latent_dim (int): Dimension của latent z
            num_layers (int): Số lớp GRU
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # GRU để encode chuỗi pose
        self.gru = nn.GRU(
            input_size=pose_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # MLP kết hợp GRU output và context
        combined_dim = hidden_dim + context_dim
        self.fc_hidden = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(
        self, 
        pose: torch.Tensor, 
        context: torch.Tensor,
        pose_mask: torch.Tensor = None
    ) -> tuple:
        """
        Encode pose sequence thành phân phối posterior.
        
        Args:
            pose: Tensor shape [batch, seq_len, pose_dim]
            context: Tensor shape [batch, context_dim]
            pose_mask: Tensor shape [batch, seq_len] (optional)
            
        Returns:
            mu: Mean [batch, latent_dim]
            logvar: Log-variance [batch, latent_dim]
        """
        # GRU encoding: output shape [batch, seq_len, hidden_dim]
        gru_out, _ = self.gru(pose)
        
        # Lấy hidden state cuối cùng (hoặc mask-aware pooling)
        if pose_mask is not None:
            # Lấy vị trí valid cuối cùng của mỗi sequence
            lengths = pose_mask.sum(dim=1).long()  # [batch]
            batch_size = pose.size(0)
            last_hidden = gru_out[torch.arange(batch_size), lengths - 1]  # [batch, hidden_dim]
        else:
            last_hidden = gru_out[:, -1, :]  # [batch, hidden_dim]
        
        # Concatenate với context
        combined = torch.cat([last_hidden, context], dim=-1)  # [batch, hidden_dim + context_dim]
        
        # MLP
        h = self.fc_hidden(combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z