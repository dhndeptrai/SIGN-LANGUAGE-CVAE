"""
Main training script cho Sign Language CVAE - Phiên bản nâng cấp (Tuần 3).

Cải tiến so với base:
    1. KL Annealing: Tăng lambda_kl từ 0 -> target theo lịch (tránh posterior collapse).
    2. Cosine LR Scheduler: Giảm learning rate mượt mà theo cos annealing.
    3. Early Stopping: Dừng sớm nếu val_mse không cải thiện sau N epochs.
    4. Training History: Lưu toàn bộ metrics ra JSON để vẽ đồ thị.
    5. Teacher Forcing Decay: Giảm dần teacher forcing ratio theo epoch.

Tham khảo:
    - Sohn et al. (2015): CVAE gốc. https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html
    - Bowman et al. (2016): KL Annealing cho VAE language models. https://arxiv.org/abs/1511.06349
    - Loshchilov & Hutter (2017): SGDR Cosine Annealing. https://arxiv.org/abs/1608.03983
"""

import os
import json
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.vocabulary import GlossVocabulary
from src.data.dataset import PhoenixDataset
from src.models.cvae import SignLanguageCVAE
from src.losses.reconstruction import PoseReconstructionLoss
from src.losses.kl_divergence import KLDivergenceLoss
from src.losses.velocity import VelocityLoss
from training.config import TrainingConfig


def compute_kl_weight(
    epoch: int,
    warmup_epochs: int,
    target_weight: float,
    anneal_type: str = "linear"
) -> float:
    """
    Tính KL weight theo lịch annealing để tránh posterior collapse.

    Trong giai đoạn warmup (epoch <= warmup_epochs), KL weight = 0 để mô hình
    tập trung học reconstruction trước. Sau warmup, weight tăng dần đến
    target_weight theo kiểu tuyến tính hoặc sigmoid.

    Kỹ thuật này được đề xuất bởi Bowman et al. (2016) cho VAE language models,
    chứng minh tránh được hiện tượng KL vanishing (posterior = prior từ đầu).

    Args:
        epoch (int): Epoch hiện tại (bắt đầu từ 1).
        warmup_epochs (int): Số epoch giữ KL = 0 trước khi bắt đầu annealing.
        target_weight (float): Giá trị KL weight cuối cùng (thường = lambda_kl).
        anneal_type (str): Kiểu annealing - "linear" hoặc "sigmoid".
            - "linear": Tăng tuyến tính từ 0 đến target.
            - "sigmoid": Tăng theo dạng S-curve (mượt hơn ở hai đầu).

    Returns:
        float: KL weight cho epoch hiện tại, nằm trong [0, target_weight].

    Example:
        >>> # Epoch 5, warmup=10, target=0.01 => weight = 0.0 (còn trong warmup)
        >>> compute_kl_weight(5, 10, 0.01)
        0.0
        >>> # Epoch 15, warmup=10 => đã qua warmup, tăng dần
        >>> compute_kl_weight(15, 10, 0.01, "linear")
        0.005
    """
    if epoch <= warmup_epochs:
        return 0.0

    progress = (epoch - warmup_epochs) / max(warmup_epochs, 1)
    progress = min(progress, 1.0)

    if anneal_type == "sigmoid":
        x = (progress - 0.5) * 10
        ratio = 1.0 / (1.0 + math.exp(-x))
    else:
        ratio = progress

    return target_weight * ratio


def compute_teacher_forcing_ratio(
    epoch: int,
    total_epochs: int,
    initial_ratio: float = 0.5,
    final_ratio: float = 0.0
) -> float:
    """
    Tính tỷ lệ teacher forcing theo lịch giảm dần (scheduled sampling).

    Teacher forcing dùng ground truth của bước t-1 làm input cho bước t,
    giúp training ổn định ở giai đoạn đầu. Việc giảm dần tỷ lệ này giúp
    mô hình dần học cách sử dụng dự đoán của chính nó (tránh exposure bias).

    Kỹ thuật này được đề xuất bởi Bengio et al. (2015) - Scheduled Sampling
    for Sequence Prediction with Recurrent Neural Networks.

    Args:
        epoch (int): Epoch hiện tại (bắt đầu từ 1).
        total_epochs (int): Tổng số epoch training.
        initial_ratio (float): Tỷ lệ teacher forcing ban đầu (epoch 1).
                               Mặc định 0.5 = 50% dùng ground truth.
        final_ratio (float): Tỷ lệ teacher forcing cuối cùng.
                             Mặc định 0.0 = hoàn toàn dùng prediction.

    Returns:
        float: Tỷ lệ teacher forcing trong [final_ratio, initial_ratio].

    Example:
        >>> # Epoch 1/50, initial=0.5, final=0.0
        >>> compute_teacher_forcing_ratio(1, 50, 0.5, 0.0)
        0.5
        >>> # Epoch 50/50
        >>> compute_teacher_forcing_ratio(50, 50, 0.5, 0.0)
        0.0
    """
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    return initial_ratio + (final_ratio - initial_ratio) * progress


class EarlyStopping:
    """
    Cơ chế Early Stopping theo validation MSE để tránh overfitting.

    Dừng training sớm khi val_mse không cải thiện sau 'patience' epoch
    liên tiếp. Checkpoint tốt nhất được lưu riêng bởi training loop.

    Attributes:
        patience (int): Số epoch chờ trước khi dừng nếu không có cải thiện.
        min_delta (float): Ngưỡng tối thiểu để coi là "cải thiện".
        counter (int): Đếm số epoch không có cải thiện liên tiếp.
        best_score (float): Score tốt nhất từ trước đến nay.
        should_stop (bool): Cờ báo hiệu cần dừng training.

    Example:
        >>> es = EarlyStopping(patience=5, min_delta=1e-4)
        >>> for val_mse in [0.5, 0.4, 0.42, 0.41, 0.41, 0.41, 0.41]:
        ...     if es.step(val_mse):
        ...         print("Early stop!")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4
    ):
        """
        Args:
            patience (int): Số epoch tối đa không cải thiện trước khi dừng.
            min_delta (float): Cải thiện tối thiểu để counter được reset về 0.
        """
        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.best_score = float('inf')
        self.should_stop = False

    def step(self, current_score: float) -> bool:
        """
        Cập nhật trạng thái early stopping với score của epoch hiện tại.

        So sánh current_score với best_score. Nếu cải thiện đủ ngưỡng
        min_delta, reset counter. Ngược lại tăng counter và kiểm tra
        có vượt patience không.

        Args:
            current_score (float): Val MSE của epoch hiện tại (thấp hơn = tốt hơn).

        Returns:
            bool: True nếu cần dừng training, False nếu tiếp tục.
        """
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: SignLanguageCVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    kl_weight: float,
    teacher_forcing_ratio: float
) -> dict:
    """
    Chạy một epoch training đầy đủ qua toàn bộ training set.

    Mỗi batch tính 3 thành phần loss:
        - MSE: Độ chính xác tọa độ pose.
        - KL: Regularize latent space (nhân với kl_weight để annealing).
        - Velocity: Smooth chuyển động giữa các frame.

    Gradient clipping (max_norm=1.0) được áp dụng để tránh gradient explosion,
    đây là kỹ thuật quan trọng khi training RNN/GRU-based models.

    Args:
        model (SignLanguageCVAE): Mô hình CVAE đang training.
        dataloader (DataLoader): DataLoader của tập train.
        optimizer (torch.optim.Optimizer): Adam optimizer đã khởi tạo.
        config (TrainingConfig): Cấu hình training (lambda_mse, lambda_vel, device).
        epoch (int): Epoch hiện tại (dùng cho progress bar).
        kl_weight (float): Trọng số KL sau annealing (tăng dần theo epoch).
        teacher_forcing_ratio (float): Tỷ lệ teacher forcing hiện tại.

    Returns:
        dict: Dictionary chứa các metrics trung bình của epoch:
            - 'loss': Tổng loss có trọng số.
            - 'mse': MSE reconstruction loss.
            - 'kl': KL divergence.
            - 'vel': Velocity smoothness loss.
            - 'kl_weight': KL weight đã dùng (để log).
            - 'tf_ratio': Teacher forcing ratio đã dùng (để log).
    """
    model.train()

    mse_loss_fn = PoseReconstructionLoss()
    kl_loss_fn = KLDivergenceLoss()
    vel_loss_fn = VelocityLoss()

    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    total_vel = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=(
            f"Epoch {epoch:03d} "
            f"[Train] "
            f"KL={kl_weight:.4f} "
            f"TF={teacher_forcing_ratio:.2f}"
        )
    )

    for batch in progress_bar:

        text_ids = batch['text_ids'].to(config.device)
        text_mask = batch['text_mask'].to(config.device)

        pose = batch['pose'].to(config.device)
        pose_mask = batch['pose_mask'].to(config.device)

        outputs = model(
            text_ids=text_ids,
            text_mask=text_mask,
            target_pose=pose,
            pose_mask=pose_mask,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        pose_pred = outputs['pose_pred']

        mse_loss = mse_loss_fn(pose_pred, pose, pose_mask)
        kl_loss = kl_loss_fn(
            outputs['mu_post'], outputs['logvar_post'],
            outputs['mu_prior'], outputs['logvar_prior']
        )
        vel_loss = vel_loss_fn(pose_pred, pose_mask)

        loss = (
            config.lambda_mse * mse_loss
            + kl_weight * kl_loss
            + config.lambda_vel * vel_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        'vel': total_vel / n,
        'kl_weight': kl_weight,
        'tf_ratio': teacher_forcing_ratio
    }


def validate(
    model: SignLanguageCVAE,
    dataloader: DataLoader,
    config: TrainingConfig
) -> dict:
    """
    Chạy validation theo đúng quy trình CVAE inference.

    Validation KHÔNG dùng posterior encoder (chỉ dùng khi training).
    Mô hình sample z từ prior p(z|c) rồi decode, giống như lúc inference thực tế.
    KL loss KHÔNG được tính vì không có posterior.

    Teacher forcing ratio = 0.0 trong validation để đánh giá khả năng
    thực sự của mô hình khi tự sinh ra chuỗi.

    Args:
        model (SignLanguageCVAE): Mô hình CVAE đang evaluate.
        dataloader (DataLoader): DataLoader của tập validation (dev).
        config (TrainingConfig): Cấu hình (device).

    Returns:
        dict: Dictionary chứa metrics trung bình:
            - 'mse': MSE giữa pose predicted và ground truth.
            - 'vel': Velocity smoothness của pose predicted.
    """
    model.eval()

    mse_loss_fn = PoseReconstructionLoss()
    vel_loss_fn = VelocityLoss()

    total_mse = 0.0
    total_vel = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Validation]", leave=False):

            text_ids = batch['text_ids'].to(config.device)
            text_mask = batch['text_mask'].to(config.device)

            pose = batch['pose'].to(config.device)
            pose_mask = batch['pose_mask'].to(config.device)

            # Inference mode: target_pose=None => sample from prior
            outputs = model(
                text_ids=text_ids,
                text_mask=text_mask,
                target_pose=None,
                pose_mask=pose_mask,
                teacher_forcing_ratio=0.0
            )

            pose_pred = outputs['pose_pred']

            total_mse += mse_loss_fn(pose_pred, pose, pose_mask).item()
            total_vel += vel_loss_fn(pose_pred, pose_mask).item()

    n = len(dataloader)
    return {
        'mse': total_mse / n,
        'vel': total_vel / n
    }


def main():
    """
    Entry point: khởi tạo toàn bộ components và chạy training loop.

    Pipeline:
        1. Load config, vocab, dataset.
        2. Khởi tạo model, optimizer, scheduler, early stopping.
        3. Lặp qua từng epoch: train -> validate -> save checkpoint.
        4. Lưu training history ra JSON để vẽ đồ thị sau.
    """
    config = TrainingConfig()

    print("=" * 60)
    print("Loading vocabulary...")

    vocab = GlossVocabulary.load(config.vocab_path)
    print(f"Vocabulary size: {len(vocab)}")

    print("\nLoading datasets...")

    train_dataset = PhoenixDataset(
        config.train_h5, vocab, config.max_text_len, config.max_pose_len
    )
    dev_dataset = PhoenixDataset(
        config.dev_h5, vocab, config.max_text_len, config.max_pose_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(
        f"Train samples: {len(train_dataset)} | "
        f"Validation samples: {len(dev_dataset)}"
    )

    print("\nInitializing model...")

    model = SignLanguageCVAE(
        vocab_size=len(vocab),
        d_model=config.d_model,
        latent_dim=config.latent_dim,
        pose_dim=config.pose_dim,
        max_text_len=config.max_text_len,
        max_pose_len=config.max_pose_len
    ).to(config.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    print(f"Device: {config.device}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.01
    )

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience, min_delta=1e-4
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    best_val_mse = float('inf')
    training_history = []

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, config.num_epochs + 1):

        kl_weight = compute_kl_weight(
            epoch=epoch,
            warmup_epochs=config.kl_warmup_epochs,
            target_weight=config.lambda_kl,
            anneal_type="linear"
        )

        tf_ratio = compute_teacher_forcing_ratio(
            epoch=epoch,
            total_epochs=config.num_epochs,
            initial_ratio=config.teacher_forcing_ratio,
            final_ratio=0.0
        )

        train_metrics = train_one_epoch(
            model=model, dataloader=train_loader,
            optimizer=optimizer, config=config,
            epoch=epoch, kl_weight=kl_weight,
            teacher_forcing_ratio=tf_ratio
        )

        val_metrics = validate(model=model, dataloader=dev_loader, config=config)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch:03d}/{config.num_epochs}")
        print(f"LR={current_lr:.2e} | KL_w={kl_weight:.4f} | TF={tf_ratio:.2f}")
        print(
            f"Train -> loss={train_metrics['loss']:.4f} | "
            f"mse={train_metrics['mse']:.4f} | "
            f"kl={train_metrics['kl']:.4f} | "
            f"vel={train_metrics['vel']:.4f}"
        )
        print(f"Val   -> mse={val_metrics['mse']:.4f} | vel={val_metrics['vel']:.4f}")

        epoch_record = {
            'epoch': epoch,
            'lr': current_lr,
            'kl_weight': kl_weight,
            'tf_ratio': tf_ratio,
            'train': train_metrics,
            'val': val_metrics
        }
        training_history.append(epoch_record)

        history_path = os.path.join(config.checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_mse': best_val_mse,
                'config': config.__dict__
            }, best_path)
            print(f"✅ Best model saved! Val MSE: {best_val_mse:.4f}")

        if epoch % config.save_every == 0:
            ckpt_path = os.path.join(
                config.checkpoint_dir, f"cvae_epoch_{epoch:03d}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_mse': val_metrics['mse']
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

        if early_stopping.step(val_metrics['mse']):
            print(f"\n⏹ Early stopping triggered at epoch {epoch}")
            break

    print("\n🎉 Training completed!")
    print(f"Best validation MSE: {best_val_mse:.4f}")


if __name__ == "__main__":
    main()
