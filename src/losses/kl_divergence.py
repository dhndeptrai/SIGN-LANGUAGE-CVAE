"""
KL Divergence Loss giữa posterior và prior.

KL(q(z|x,c) || p(z|c)) được tính theo công thức closed-form của 2 Gaussian.
"""

import torch
import torch.nn as nn


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence giữa 2 phân phối Gaussian.
    
    KL(N(mu1, sigma1^2) || N(mu2, sigma2^2))
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
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor
    ) -> torch.Tensor:
        """
        Tính KL divergence.
        
        Args:
            mu_post, logvar_post: Posterior params [batch, latent_dim]
            mu_prior, logvar_prior: Prior params [batch, latent_dim]
            
        Returns:
            kl_loss: Scalar tensor
        """
        # Công thức KL cho 2 Gaussian:
        # KL = 0.5 * sum(sigma_post^2 / sigma_prior^2 + (mu_prior - mu_post)^2 / sigma_prior^2 - 1 + log(sigma_prior^2 / sigma_post^2))
        
        var_post = torch.exp(logvar_post)
        var_prior = torch.exp(logvar_prior)
        
        kl = 0.5 * (
            var_post / var_prior
            + ((mu_prior - mu_post) ** 2) / var_prior
            - 1
            + logvar_prior - logvar_post
        )
        
        # Sum over latent dimensions
        kl = kl.sum(dim=-1)  # [batch]
        
        if self.reduction == 'mean':
            return kl.mean()
        else:
            return kl.sum()