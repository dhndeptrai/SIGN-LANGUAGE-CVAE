"""
Configuration file cho training.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters cho training."""
    
    # Data paths
    train_h5 = "data/processed/train_data.h5"
    dev_h5   = "data/processed/dev_data.h5"
    vocab_path = "data/vocabulary/gloss_vocab.pkl"
    
    # Model architecture
    d_model: int = 512
    latent_dim: int = 128
    pose_dim: int = 225
    max_text_len: int = 100
    max_pose_len: int = 300
    
    # Training
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    teacher_forcing_ratio: float = 0.5
    
    # Loss weights
    lambda_mse: float = 1.0
    lambda_kl: float = 0.01  # KL weight (beta trong beta-VAE)
    lambda_vel: float = 0.1
    
    # Optimizer
    weight_decay: float = 1e-5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save every N epochs
    
    # Device
    device: str = "cuda"  # hoặc "cpu"