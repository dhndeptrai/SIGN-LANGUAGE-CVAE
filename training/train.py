"""
Main training script cho Sign Language CVAE.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.vocabulary import GlossVocabulary
from src.data.dataset import PhoenixDataset
from src.models.cvae import SignLanguageCVAE
from src.losses.reconstruction import PoseReconstructionLoss
from src.losses.kl_divergence import KLDivergenceLoss
from src.losses.velocity import VelocityLoss
from training.config import TrainingConfig


def train_one_epoch(model, dataloader, optimizer, config, epoch):
    model.train()

    mse_loss_fn = PoseReconstructionLoss()
    kl_loss_fn = KLDivergenceLoss()
    vel_loss_fn = VelocityLoss()

    total_loss, total_mse, total_kl, total_vel = 0, 0, 0, 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        text_ids = batch['text_ids'].to(config.device)
        text_mask = batch['text_mask'].to(config.device)
        pose = batch['pose'].to(config.device)
        pose_mask = batch['pose_mask'].to(config.device)

        outputs = model(
            text_ids=text_ids,
            text_mask=text_mask,
            target_pose=pose,  # ✅ training dùng posterior
            pose_mask=pose_mask,
            teacher_forcing_ratio=config.teacher_forcing_ratio
        )

        pose_pred = outputs['pose_pred']
        mu_prior = outputs['mu_prior']
        logvar_prior = outputs['logvar_prior']
        mu_post = outputs['mu_post']
        logvar_post = outputs['logvar_post']

        mse_loss = mse_loss_fn(pose_pred, pose, pose_mask)
        kl_loss = kl_loss_fn(mu_post, logvar_post, mu_prior, logvar_prior)
        vel_loss = vel_loss_fn(pose_pred, pose_mask)

        loss = (
            config.lambda_mse * mse_loss +
            config.lambda_kl * kl_loss +
            config.lambda_vel * vel_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_kl += kl_loss.item()
        total_vel += vel_loss.item()

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{mse_loss.item():.4f}",
            'kl': f"{kl_loss.item():.4f}",
            'vel': f"{vel_loss.item():.4f}"
        })

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'mse': total_mse / n,
        'kl': total_kl / n,
        'vel': total_vel / n
    }


def validate(model, dataloader, config):
    """
    Validation đúng lý thuyết CVAE:
    - KHÔNG dùng posterior
    - KHÔNG có KL
    """
    model.eval()

    mse_loss_fn = PoseReconstructionLoss()
    vel_loss_fn = VelocityLoss()

    total_mse, total_vel = 0.0, 0.0
    n = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            text_ids = batch['text_ids'].to(config.device)
            text_mask = batch['text_mask'].to(config.device)
            pose = batch['pose'].to(config.device)
            pose_mask = batch['pose_mask'].to(config.device)

            outputs = model(
                text_ids=text_ids,
                text_mask=text_mask,
                target_pose=None,  # ✅ chỉ dùng prior p(z|c)
                pose_mask=pose_mask,
                teacher_forcing_ratio=0.0
            )

            mse_loss = mse_loss_fn(outputs['pose_pred'], pose, pose_mask)
            vel_loss = vel_loss_fn(outputs['pose_pred'], pose_mask)

            total_mse += mse_loss.item()
            total_vel += vel_loss.item()

    return {
        'mse': total_mse / n,
        'vel': total_vel / n
    }


def main():
    config = TrainingConfig()

    print("Loading vocabulary...")
    vocab = GlossVocabulary.load(config.vocab_path)

    print("Loading datasets...")
    train_dataset = PhoenixDataset(config.train_h5, vocab, config.max_text_len, config.max_pose_len)
    dev_dataset = PhoenixDataset(config.dev_h5, vocab, config.max_text_len, config.max_pose_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print("Initializing model...")
    model = SignLanguageCVAE(
        vocab_size=len(vocab),
        d_model=config.d_model,
        latent_dim=config.latent_dim,
        pose_dim=config.pose_dim,
        max_text_len=config.max_text_len,
        max_pose_len=config.max_pose_len
    ).to(config.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    best_val_mse = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*60}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, config, epoch)

        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"MSE: {train_metrics['mse']:.4f} | "
              f"KL: {train_metrics['kl']:.4f} | "
              f"Vel: {train_metrics['vel']:.4f}")

        val_metrics = validate(model, dev_loader, config)

        print(f"Val MSE: {val_metrics['mse']:.4f} | "
              f"Val Vel: {val_metrics['vel']:.4f}")

        # ✅ Save best model
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_path = os.path.join(config.checkpoint_dir, "best_model.pt")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': best_val_mse
            }, best_path)

            print(f"🌟 Best model saved! Val MSE: {best_val_mse:.4f}")

        # Save periodic checkpoint
        if epoch % config.save_every == 0:
            path = os.path.join(config.checkpoint_dir, f"cvae_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, path)
            print(f"💾 Checkpoint saved: {path}")

    print("\n🎉 Training completed!")


if __name__ == "__main__":
    main()