"""
Conditional Variational Autoencoder (CVAE) cho Sign Language Production.

Tham khảo:
- Sohn et al. (2015): Learning Structured Output Representation using Deep CVAE
- Stoll et al. (2019): Text2Sign - Sign Language Production using Neural Machine Translation
"""

import torch
import torch.nn as nn
from .text_encoder import TransformerTextEncoder
from .prior_network import PriorNetwork
from .posterior_encoder import PosteriorEncoder
from .pose_decoder import PoseDecoder


class SignLanguageCVAE(nn.Module):
    """
    CVAE model cho Sign Language Production.
    
    Components:
        1. Text Encoder: Encode glosses -> context c
        2. Prior Network: p(z|c)
        3. Posterior Encoder: q(z|x,c) [training only]
        4. Pose Decoder: p(x|z,c)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        latent_dim: int = 128,
        pose_dim: int = 225,
        max_text_len: int = 100,
        max_pose_len: int = 300
    ):
        """
        Args:
            vocab_size (int): Kích thước vocabulary
            d_model (int): Dimension của Transformer
            latent_dim (int): Dimension của latent z
            pose_dim (int): Dimension của pose (225 cho MediaPipe)
            max_text_len (int): Độ dài tối đa input text
            max_pose_len (int): Độ dài tối đa output pose
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.pose_dim = pose_dim
        
        # 1. Text Encoder
        self.text_encoder = TransformerTextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_text_len
        )
        
        # 2. Prior Network
        self.prior = PriorNetwork(
            context_dim=d_model,
            latent_dim=latent_dim
        )
        
        # 3. Posterior Encoder
        self.posterior = PosteriorEncoder(
            pose_dim=pose_dim,
            context_dim=d_model,
            latent_dim=latent_dim
        )
        
        # 4. Pose Decoder
        self.decoder = PoseDecoder(
            latent_dim=latent_dim,
            context_dim=d_model,
            pose_dim=pose_dim,
            max_len=max_pose_len
        )
    
    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        target_pose: torch.Tensor = None,
        pose_mask: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5
    ) -> dict:
        """
        Forward pass của CVAE.
        
        Args:
            text_ids: [batch, text_len]
            text_mask: [batch, text_len]
            target_pose: [batch, pose_len, pose_dim] (optional, for training)
            pose_mask: [batch, pose_len] (optional)
            teacher_forcing_ratio: Tỷ lệ teacher forcing
            
        Returns:
            Dict chứa:
                - pose_pred: Predicted pose sequence
                - mu_prior, logvar_prior: Prior distribution params
                - mu_post, logvar_post: Posterior distribution params (if training)
                - z: Sampled latent
        """
        # Encode text -> context
        context = self.text_encoder(text_ids, text_mask)  # [batch, d_model]
        
        # Prior distribution p(z|c)
        mu_prior, logvar_prior = self.prior(context)
        
        # Training mode: use posterior q(z|x,c)
        if self.training and target_pose is not None:
            mu_post, logvar_post = self.posterior(target_pose, context, pose_mask)
            z = self.posterior.sample(mu_post, logvar_post)
        else:
            # Inference mode: sample from prior
            mu_post, logvar_post = None, None
            z = self.prior.sample(mu_prior, logvar_prior)
        
        # Decode z -> pose
        pose_pred = self.decoder(z, context, target_pose, teacher_forcing_ratio)
        
        return {
            'pose_pred': pose_pred,
            'mu_prior': mu_prior,
            'logvar_prior': logvar_prior,
            'mu_post': mu_post,
            'logvar_post': logvar_post,
            'z': z
        }
    
    def generate(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Sinh pose sequence từ text (inference mode).
        
        Args:
            text_ids: [batch, text_len]
            text_mask: [batch, text_len]
            num_samples: Số lượng mẫu sinh ra cho mỗi input
            
        Returns:
            pose_samples: [batch * num_samples, pose_len, pose_dim]
        """
        self.eval()
        with torch.no_grad():
            batch_size = text_ids.size(0)
            
            # Encode text
            context = self.text_encoder(text_ids, text_mask)  # [batch, d_model]
            
            # Sample z từ prior
            mu_prior, logvar_prior = self.prior(context)
            
            all_samples = []
            for _ in range(num_samples):
                z = self.prior.sample(mu_prior, logvar_prior)
                pose_pred = self.decoder(z, context, target_pose=None, teacher_forcing_ratio=0.0)
                all_samples.append(pose_pred)
            
            return torch.cat(all_samples, dim=0)