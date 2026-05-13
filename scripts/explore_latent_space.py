"""
Yêu cầu nâng cao (Tuần 4): Trực quan hóa và khám phá Latent Space.

Script này thực hiện 2 nhiệm vụ:
    1. Diversity Sampling: Từ cùng 1 câu gloss, sinh ra N biến thể pose
       bằng cách sample z khác nhau từ prior p(z|c). Lưu từng biến thể
       ra .npy để M2 render thành video.

    2. Latent Space Visualization: Thu thập vector z từ nhiều câu, chiếu
       xuống 2D bằng t-SNE, vẽ biểu đồ scatter để kiểm tra cấu trúc
       của không gian tiềm ẩn.

Cách dùng:
    # Sinh 5 biến thể từ 1 câu:
    python scripts/explore_latent_space.py \
        --mode diversity \
        --checkpoint checkpoints/best_model.pt \
        --gloss "MORGEN SONNE SCHEINEN" \
        --num_samples 5 \
        --output_dir outputs/diversity/

    # Visualize latent space của toàn bộ val set:
    python scripts/explore_latent_space.py \
        --mode tsne \
        --checkpoint checkpoints/best_model.pt \
        --output_dir outputs/latent_viz/

Tham khảo:
    - van der Maaten & Hinton (2008): Visualizing High-Dimensional Data
      using t-SNE. https://jmlr.org/papers/v9/vandermaaten08a.html
    - Sohn et al. (2015): CVAE diversity sampling.
"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (Colab/server compatible)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

from src.data.vocabulary import GlossVocabulary
from src.data.dataset import PhoenixDataset
from src.models.cvae import SignLanguageCVAE
from training.config import TrainingConfig


# ============================================================
# Helper: Load model
# ============================================================

def load_model(
    checkpoint_path: str,
    vocab_size: int,
    config: TrainingConfig,
    device: torch.device
) -> SignLanguageCVAE:
    """
    Load mô hình CVAE từ checkpoint và chuyển sang eval mode.

    Args:
        checkpoint_path (str): Đường dẫn file checkpoint .pt.
        vocab_size (int): Kích thước vocabulary.
        config (TrainingConfig): Cấu hình kiến trúc.
        device (torch.device): Device đích.

    Returns:
        SignLanguageCVAE: Mô hình đã load weights ở eval mode.
    """
    model = SignLanguageCVAE(
        vocab_size=vocab_size,
        d_model=config.d_model,
        latent_dim=config.latent_dim,
        pose_dim=config.pose_dim,
        max_text_len=config.max_text_len,
        max_pose_len=config.max_pose_len
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded from epoch {checkpoint.get('epoch', '?')}, "
          f"val_mse={checkpoint.get('val_mse', '?'):.4f}")
    return model


# ============================================================
# Mode 1: Diversity Sampling
# ============================================================

def run_diversity_sampling(
    model: SignLanguageCVAE,
    gloss_sentence: str,
    vocab: GlossVocabulary,
    config: TrainingConfig,
    device: torch.device,
    num_samples: int,
    output_dir: str
) -> None:
    """
    Sinh nhiều biến thể pose từ cùng 1 câu gloss bằng cách sample z khác nhau.

    Đây là yêu cầu nâng cao của đề bài: chứng minh rằng CVAE sinh được
    các biến thể chuyển động đa dạng (khác tốc độ, biên độ) từ cùng 1 input.

    Quy trình:
        context c = Encode(gloss)
        mu_prior, logvar_prior = Prior(c)
        for i in range(num_samples):
            z_i = sample(mu_prior, logvar_prior)   # z khác nhau mỗi lần
            pose_i = Decoder(z_i, c)
            save(pose_i)

    Args:
        model (SignLanguageCVAE): Mô hình đã load.
        gloss_sentence (str): Câu gloss đầu vào, các token cách nhau bởi space.
        vocab (GlossVocabulary): Vocabulary.
        config (TrainingConfig): Cấu hình model.
        device (torch.device): Device.
        num_samples (int): Số lượng biến thể cần sinh.
        output_dir (str): Thư mục lưu kết quả.

    Returns:
        None. Kết quả được lưu ra file .npy và metadata .json.

    Note:
        Các file .npy được đặt tên theo pattern:
        `{gloss}_sample{i+1:02d}.npy`
    """
    from scripts.run_inference import tokenize_gloss

    tokens, mask = tokenize_gloss(gloss_sentence, vocab, config.max_text_len)
    tokens = tokens.to(device)
    mask = mask.to(device)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = gloss_sentence.replace(" ", "_")[:30]

    metadata = {
        "gloss": gloss_sentence,
        "num_samples": num_samples,
        "samples": []
    }

    print(f"\nGenerating {num_samples} diversity samples for: '{gloss_sentence}'")

    with torch.no_grad():
        # Encode text một lần dùng chung
        context = model.text_encoder(tokens, mask)           # [1, d_model]
        mu_prior, logvar_prior = model.prior(context)        # [1, latent_dim]

        for i in range(num_samples):
            # Sample z khác nhau mỗi lần từ cùng prior
            z = model.prior.sample(mu_prior, logvar_prior)   # [1, latent_dim]
            pose_pred = model.decoder(z, context, target_pose=None, teacher_forcing_ratio=0.0)

            pose_np = pose_pred.squeeze(0).cpu().numpy()     # [seq_len, pose_dim]

            # Tính velocity trung bình (đo độ "nhanh" của chuyển động)
            velocity_magnitude = np.linalg.norm(
                np.diff(pose_np, axis=0), axis=-1
            ).mean()

            # Tính spread (đo biên độ chuyển động)
            spread = pose_np.std()

            fname = f"{safe_name}_sample{i+1:02d}.npy"
            fpath = os.path.join(output_dir, fname)
            np.save(fpath, pose_np)

            sample_info = {
                "sample_id": i + 1,
                "filename": fname,
                "shape": list(pose_np.shape),
                "avg_velocity_magnitude": float(velocity_magnitude),
                "pose_spread": float(spread)
            }
            metadata["samples"].append(sample_info)

            print(
                f"  Sample {i+1:02d}: shape={pose_np.shape} | "
                f"avg_vel={velocity_magnitude:.4f} | spread={spread:.4f}"
            )

    # Lưu metadata
    meta_path = os.path.join(output_dir, f"{safe_name}_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    print("Done! Chuyển các file .npy này cho M2 để render thành video.")


# ============================================================
# Mode 2: t-SNE Latent Space Visualization
# ============================================================

def collect_latent_vectors(
    model: SignLanguageCVAE,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 500
) -> tuple:
    """
    Thu thập vector z từ prior của nhiều câu trong dataset.

    Dùng để visualize không gian tiềm ẩn: nếu không gian có cấu trúc,
    các câu có nghĩa tương đồng sẽ cluster gần nhau.

    Args:
        model (SignLanguageCVAE): Mô hình CVAE ở eval mode.
        dataloader (DataLoader): DataLoader của val/test set.
        device (torch.device): Device.
        max_samples (int): Số lượng mẫu tối đa để thu thập (tránh OOM).

    Returns:
        tuple:
            - z_vectors (np.ndarray): Shape [N, latent_dim]. Các vector z thu thập được.
            - mu_vectors (np.ndarray): Shape [N, latent_dim]. Mean của prior (ổn định hơn z).
            - text_lengths (list[int]): Độ dài câu gloss của từng mẫu.
    """
    z_list = []
    mu_list = []
    text_len_list = []
    total_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if total_collected >= max_samples:
                break

            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)

            context = model.text_encoder(text_ids, text_mask)
            mu_prior, logvar_prior = model.prior(context)
            z = model.prior.sample(mu_prior, logvar_prior)

            z_list.append(z.cpu().numpy())
            mu_list.append(mu_prior.cpu().numpy())

            # Lấy độ dài câu (số token valid)
            lengths = text_mask.sum(dim=1).cpu().numpy().tolist()
            text_len_list.extend(lengths)

            total_collected += text_ids.size(0)

    z_vectors = np.concatenate(z_list, axis=0)[:max_samples]
    mu_vectors = np.concatenate(mu_list, axis=0)[:max_samples]
    text_lengths = text_len_list[:max_samples]

    return z_vectors, mu_vectors, text_lengths


def plot_tsne(
    vectors: np.ndarray,
    color_values: list,
    title: str,
    output_path: str,
    perplexity: int = 30
) -> None:
    """
    Chiếu vector xuống 2D bằng t-SNE và vẽ scatter plot.

    Args:
        vectors (np.ndarray): Shape [N, dim]. Dữ liệu cần visualize.
        color_values (list): Giá trị dùng để tô màu từng điểm (ví dụ: text length).
        title (str): Tiêu đề của biểu đồ.
        output_path (str): Đường dẫn lưu file ảnh .png.
        perplexity (int): Tham số t-SNE (thường 5-50, mặc định 30).

    Returns:
        None. Kết quả được lưu ra file ảnh.

    Note:
        t-SNE là một thuật toán non-linear dimensionality reduction.
        Tham khảo: van der Maaten & Hinton (2008).
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn không có sẵn. Cài: pip install scikit-learn")
        return

    print(f"  Running t-SNE on {vectors.shape[0]} vectors (dim={vectors.shape[1]})...")

    # t-SNE thực sự chạy ở đây
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000
    )
    z_2d = tsne.fit_transform(vectors)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=color_values,
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    plt.colorbar(scatter, ax=ax, label='Text Length (# tokens)')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def run_tsne_visualization(
    model: SignLanguageCVAE,
    vocab: GlossVocabulary,
    config: TrainingConfig,
    device: torch.device,
    output_dir: str,
    max_samples: int = 300
) -> None:
    """
    Thu thập z vectors từ val set và tạo biểu đồ t-SNE.

    Args:
        model (SignLanguageCVAE): Mô hình CVAE.
        vocab (GlossVocabulary): Vocabulary.
        config (TrainingConfig): Cấu hình.
        device (torch.device): Device.
        output_dir (str): Thư mục lưu biểu đồ .png.
        max_samples (int): Số lượng mẫu tối đa.

    Returns:
        None.
    """
    print(f"\nCollecting latent vectors from val set (max {max_samples} samples)...")

    # Load val dataset
    dev_dataset = PhoenixDataset(config.dev_h5, vocab, config.max_text_len, config.max_pose_len)
    dev_loader = DataLoader(
        dev_dataset, batch_size=32, shuffle=False, num_workers=2
    )

    z_vectors, mu_vectors, text_lengths = collect_latent_vectors(
        model, dev_loader, device, max_samples=max_samples
    )
    print(f"  Collected {z_vectors.shape[0]} vectors of dim {z_vectors.shape[1]}")

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: t-SNE của z (sampled)
    plot_tsne(
        vectors=z_vectors,
        color_values=text_lengths,
        title="t-SNE of Sampled Latent Vectors z (colored by text length)",
        output_path=os.path.join(output_dir, "tsne_z_sampled.png"),
        perplexity=min(30, len(z_vectors) // 4)
    )

    # Plot 2: t-SNE của mu_prior (ổn định hơn z vì không có random noise)
    plot_tsne(
        vectors=mu_vectors,
        color_values=text_lengths,
        title="t-SNE of Prior Mean μ (colored by text length)",
        output_path=os.path.join(output_dir, "tsne_mu_prior.png"),
        perplexity=min(30, len(mu_vectors) // 4)
    )

    # Lưu vectors để tái sử dụng
    np.save(os.path.join(output_dir, "z_vectors.npy"), z_vectors)
    np.save(os.path.join(output_dir, "mu_vectors.npy"), mu_vectors)

    print(f"\nt-SNE visualization done! Kết quả lưu tại: {output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    """
    Entry point: parse arguments và chạy mode tương ứng.
    """
    parser = argparse.ArgumentParser(
        description="Explore CVAE latent space: diversity sampling hoặc t-SNE visualization."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["diversity", "tsne"],
        required=True,
        help="'diversity': sinh nhiều biến thể từ 1 câu. 'tsne': visualize latent space."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/vocabulary/gloss_vocab.pkl"
    )
    parser.add_argument(
        "--gloss",
        type=str,
        default=None,
        help="Câu gloss đầu vào (chỉ dùng cho mode=diversity)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Số biến thể cần sinh (mode=diversity, đề bài yêu cầu 3-5)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/latent_exploration"
    )
    parser.add_argument(
        "--max_tsne_samples",
        type=int,
        default=300,
        help="Số mẫu tối đa cho t-SNE (mode=tsne)."
    )
    args = parser.parse_args()

    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading vocabulary...")
    vocab = GlossVocabulary.load(args.vocab)

    print("Loading model...")
    model = load_model(args.checkpoint, len(vocab), config, device)

    if args.mode == "diversity":
        if args.gloss is None:
            parser.error("--gloss là bắt buộc khi mode=diversity")
        run_diversity_sampling(
            model=model,
            gloss_sentence=args.gloss,
            vocab=vocab,
            config=config,
            device=device,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )

    elif args.mode == "tsne":
        run_tsne_visualization(
            model=model,
            vocab=vocab,
            config=config,
            device=device,
            output_dir=args.output_dir,
            max_samples=args.max_tsne_samples
        )


if __name__ == "__main__":
    main()
