"""
Prior Network: Dự đoán phân phối tiềm ẩn z từ context c.

Trong CVAE, prior p(z|c) được tham số hóa bằng một MLP đơn giản.
Tham khảo: Sohn et al., 2015 - "Learning Structured Output Representation using Deep CVAE"
"""

import torch
import torch.nn as nn


class PriorNetwork(nn.Module):
    """
    Prior Network dự đoán phân phối Gaussian p(z|c) từ context vector.
    
    Output: 
        - mu: Mean của Gaussian
        - logvar: Log-variance của Gaussian
    """
    
    def __init__(self, context_dim: int = 512, latent_dim: int = 128, hidden_dim: int = 256):
        """
        Args:
            context_dim (int): Dimension của context vector từ Text Encoder
            latent_dim (int): Dimension của latent variable z
            hidden_dim (int): Dimension của hidden layer
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # MLP để tính mu và logvar
        self.fc_hidden = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, context: torch.Tensor) -> tuple:
        """
        Dự đoán tham số phân phối prior.
        
        Args:
            context: Tensor shape [batch, context_dim]
            
        Returns:
            mu: Mean vector [batch, latent_dim]
            logvar: Log-variance vector [batch, latent_dim]
        """
        h = self.fc_hidden(context)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon.
        
        Args:
            mu: Mean [batch, latent_dim]
            logvar: Log-variance [batch, latent_dim]
            
        Returns:
            z: Sampled latent [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z