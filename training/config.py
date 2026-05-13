"""
Cấu hình hyperparameter cho toàn bộ pipeline training.

Tất cả siêu tham số được tập trung ở đây để dễ điều chỉnh
và tái hiện thực nghiệm (reproducibility).
"""

import torch
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """
    Lớp cấu hình chứa tất cả siêu tham số training.

    Phân nhóm:
        - Paths: Đường dẫn file dữ liệu và checkpoint.
        - Architecture: Tham số kiến trúc mô hình.
        - Training: Tham số huấn luyện (batch size, epochs, ...).
        - Loss Weights: Lambda cho từng thành phần loss.
        - KL Annealing: Lịch tăng dần KL weight.
        - Scheduler & Regularization: LR decay, early stopping.
        - Device: CPU hay CUDA.

    Tham khảo:
        - Bowman et al. (2016) cho KL annealing scheme.
        - Higgins et al. (2017) cho beta-VAE (lambda_kl < 1).
    """

    # ---- Paths ----
    train_h5: str = "data/processed/train_data.h5"
    """Đường dẫn file HDF5 của tập train."""

    dev_h5: str = "data/processed/dev_data.h5"
    """Đường dẫn file HDF5 của tập validation."""

    test_h5: str = "data/processed/test_data.h5"
    """Đường dẫn file HDF5 của tập test."""

    vocab_path: str = "data/vocabulary/gloss_vocab.pkl"
    """Đường dẫn file vocabulary đã build."""

    checkpoint_dir: str = "checkpoints"
    """Thư mục lưu checkpoint."""

    # ---- Model Architecture ----
    d_model: int = 512
    """Dimension hidden state của Transformer và context vector c."""

    latent_dim: int = 128
    """Dimension của latent variable z trong CVAE."""

    pose_dim: int = 225
    """
    Dimension của mỗi pose frame.
    225 = 33 body landmarks * 3 (x,y,z)
        + 21 left hand * 3
        + 21 right hand * 3
    (MediaPipe Holistic output)
    """

    max_text_len: int = 100
    """Độ dài tối đa của chuỗi gloss đầu vào."""

    max_pose_len: int = 300
    """Độ dài tối đa của chuỗi pose đầu ra."""

    # ---- Training ----
    batch_size: int = 16
    """Số mẫu mỗi batch. Giảm xuống 8 nếu Colab hết RAM."""

    num_epochs: int = 50
    """Tổng số epoch tối đa (có thể dừng sớm do early stopping)."""

    learning_rate: float = 1e-4
    """Learning rate ban đầu cho Adam optimizer."""

    weight_decay: float = 1e-5
    """L2 regularization coefficient."""

    teacher_forcing_ratio: float = 0.5
    """
    Tỷ lệ teacher forcing ban đầu.
    Sẽ giảm dần về 0.0 theo epoch (scheduled sampling).
    """

    save_every: int = 5
    """Lưu checkpoint định kỳ mỗi N epoch."""

    # ---- Loss Weights ----
    lambda_mse: float = 1.0
    """Trọng số cho Reconstruction (MSE) Loss."""

    lambda_kl: float = 0.01
    """
    Trọng số KL Divergence cuối cùng (target sau annealing).
    Giữ nhỏ (< 1) để không làm át Reconstruction Loss.
    Tham khảo: Higgins et al. (2017) beta-VAE.
    """

    lambda_vel: float = 0.1
    """
    Trọng số Velocity Smoothness Loss.
    Tham khảo: Baltatzis et al. (2024) Neural Sign Actors.
    """

    # ---- KL Annealing ----
    kl_warmup_epochs: int = 10
    """
    Số epoch giữ KL weight = 0 trước khi bắt đầu tăng.
    Sau warmup, KL weight tăng tuyến tính đến lambda_kl
    trong thêm kl_warmup_epochs epoch nữa.
    Tham khảo: Bowman et al. (2016).
    """

    # ---- Scheduler & Regularization ----
    early_stopping_patience: int = 10
    """Dừng training nếu val_mse không cải thiện trong N epoch liên tiếp."""

    # ---- Device ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """
    Device chạy training: tự động phát hiện "cuda" nếu có GPU, ngược lại "cpu".
    Tương thích với môi trường Kaggle/Colab (có GPU) và máy cá nhân (CPU).
    """
