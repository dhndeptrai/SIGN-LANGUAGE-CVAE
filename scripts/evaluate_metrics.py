"""
Script đánh giá mô hình trên tập Test với đầy đủ các độ đo.

Tính toán các metrics theo yêu cầu đề bài:
    1. MSE (Mean Squared Error): Sai số tọa độ trung bình.
    2. DTW (Dynamic Time Warping): Đo độ tương đồng chuỗi thời gian.
    3. KL-Divergence (Latent Space): Kiểm tra cấu trúc không gian tiềm ẩn.

Kết quả được lưu ra file JSON và in ra console dưới dạng bảng IEEE.

Cách dùng:
    python scripts/evaluate_metrics.py \\
        --checkpoint checkpoints/best_model.pt \\
        --split test \\
        --output_dir outputs/evaluation/

Tham khảo:
    - Salvador & Chan (2007): FastDTW - Toward Accurate Dynamic Time Warping
      in Linear Time and Space. https://cs.fit.edu/~pkc/papers/tdm04.pdf
    - Müller (2007): Dynamic Time Warping. Information Retrieval for Music
      and Motion. Springer. https://doi.org/10.1007/978-3-540-74048-3_4
    - Koller et al. (2015): RWTH-PHOENIX-Weather 2014 benchmark.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Thêm root vào path để import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.vocabulary import GlossVocabulary
from src.data.dataset import PhoenixDataset
from src.models.cvae import SignLanguageCVAE
from training.config import TrainingConfig


# ============================================================
# DTW Implementation
# ============================================================

def compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Tính khoảng cách Dynamic Time Warping (DTW) giữa hai chuỗi thời gian.

    DTW tìm alignment tối ưu giữa hai chuỗi có thể có độ dài khác nhau,
    cho phép so sánh chuỗi có tốc độ biến đổi khác nhau - rất phù hợp
    cho dữ liệu Sign Language vì cùng 1 câu có thể ký hiệu nhanh/chậm.

    Độ phức tạp: O(m * n) thời gian, O(m * n) bộ nhớ.
    Với chuỗi dài (>300 frames), nên dùng FastDTW (O(n) với radius).

    Args:
        seq1 (np.ndarray): Chuỗi thứ nhất, shape [T1, D].
                           Ví dụ: pose sequence dự đoán [T1, 225].
        seq2 (np.ndarray): Chuỗi thứ hai, shape [T2, D].
                           Ví dụ: pose sequence ground truth [T2, 225].

    Returns:
        float: Khoảng cách DTW (thấp hơn = hai chuỗi tương đồng hơn).
               Được chuẩn hóa theo tổng số bước alignment.

    Example:
        >>> pred = np.random.randn(50, 225)
        >>> gt   = np.random.randn(48, 225)
        >>> dist = compute_dtw_distance(pred, gt)
        >>> print(f"DTW: {dist:.4f}")  # Khoảng 1-5 với dữ liệu chuẩn hóa

    Note:
        Hàm này dùng Euclidean distance làm local cost function.
        Các implementation khác có thể dùng cosine similarity.
    """
    t1, t2 = len(seq1), len(seq2)

    # Ma trận cost: dtw_matrix[i][j] = chi phí alignment đến (i, j)
    dtw_matrix = np.full((t1 + 1, t2 + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, t1 + 1):
        for j in range(1, t2 + 1):
            # Local cost: Euclidean distance giữa 2 frame
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            # Dynamic programming: lấy min của 3 hướng di chuyển
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    # Chuẩn hóa theo tổng số frame để so sánh công bằng
    return dtw_matrix[t1, t2] / (t1 + t2)


def compute_dtw_batch(
    pred_batch: np.ndarray,
    gt_batch: np.ndarray,
    pred_lengths: np.ndarray,
    gt_lengths: np.ndarray
) -> float:
    """
    Tính DTW trung bình trên một batch, có xét đến độ dài thực tế (bỏ qua padding).

    Args:
        pred_batch (np.ndarray): Predicted poses, shape [B, T, D].
        gt_batch (np.ndarray): Ground truth poses, shape [B, T, D].
        pred_lengths (np.ndarray): Độ dài thực của mỗi predicted sequence, shape [B].
        gt_lengths (np.ndarray): Độ dài thực của mỗi ground truth sequence, shape [B].

    Returns:
        float: DTW trung bình trên toàn batch (không bao gồm padding frames).
    """
    batch_size = pred_batch.shape[0]
    total_dtw = 0.0

    for i in range(batch_size):
        pred_len = int(pred_lengths[i])
        gt_len = int(gt_lengths[i])

        if pred_len == 0 or gt_len == 0:
            continue

        # Cắt padding trước khi tính DTW
        pred_seq = pred_batch[i, :pred_len, :]
        gt_seq = gt_batch[i, :gt_len, :]

        total_dtw += compute_dtw_distance(pred_seq, gt_seq)

    return total_dtw / batch_size


# ============================================================
# KL Divergence (Latent Space Quality)
# ============================================================

def compute_latent_kl(
    model: SignLanguageCVAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50
) -> float:
    """
    Ước lượng KL Divergence trung bình giữa posterior và prior trên dataset.

    Metric này kiểm tra xem mô hình có học được không gian tiềm ẩn có ý nghĩa
    hay bị "posterior collapse" (posterior = prior, z không mang thông tin).

    KL thấp: Posterior gần với prior => mô hình có thể bị collapse.
    KL vừa (~1-10): Không gian tiềm ẩn có cấu trúc tốt.
    KL quá cao: Mô hình không regularized tốt, generation kém đa dạng.

    Args:
        model (SignLanguageCVAE): Mô hình CVAE đã train.
        dataloader (DataLoader): DataLoader của tập dữ liệu cần evaluate.
        device (torch.device): Device chạy inference.
        max_batches (int): Số batch tối đa để tính (tránh tốn thời gian quá nhiều).

    Returns:
        float: KL Divergence trung bình (scalar).

    Note:
        KL được tính theo công thức closed-form giống trong training:
        KL(q(z|x,c) || p(z|c)) sử dụng các tham số mu và logvar.
    """
    model.eval()
    total_kl = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if num_batches >= max_batches:
                break

            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            pose = batch['pose'].to(device)
            pose_mask = batch['pose_mask'].to(device)

            # Forward với posterior (training mode) để lấy cả 2 phân phối
            model.train()  # Tạm bật train mode để posterior hoạt động
            outputs = model(
                text_ids=text_ids, text_mask=text_mask,
                target_pose=pose, pose_mask=pose_mask,
                teacher_forcing_ratio=0.0
            )
            model.eval()

            mu_post = outputs['mu_post']
            logvar_post = outputs['logvar_post']
            mu_prior = outputs['mu_prior']
            logvar_prior = outputs['logvar_prior']

            if mu_post is None:
                continue

            # KL closed-form cho 2 Gaussian
            var_post = torch.exp(logvar_post)
            var_prior = torch.exp(logvar_prior)

            kl = 0.5 * (
                var_post / var_prior
                + ((mu_prior - mu_post) ** 2) / var_prior
                - 1
                + logvar_prior - logvar_post
            ).sum(dim=-1).mean()

            total_kl += kl.item()
            num_batches += 1

    return total_kl / max(num_batches, 1)


# ============================================================
# Main Evaluation Loop
# ============================================================

def evaluate(
    model: SignLanguageCVAE,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    compute_dtw_flag: bool = True,
    dtw_max_samples: int = 100
) -> dict:
    """
    Đánh giá mô hình trên toàn bộ tập dữ liệu với tất cả metrics.

    Quy trình:
        - Inference: Sample z từ prior, decode ra pose sequence.
        - Tính MSE giữa pose dự đoán và ground truth.
        - (Tùy chọn) Tính DTW với max_samples đầu tiên (DTW tốn thời gian O(n^2)).

    Args:
        model (SignLanguageCVAE): Mô hình đã train, ở eval mode.
        dataloader (DataLoader): DataLoader của tập cần đánh giá.
        config (TrainingConfig): Cấu hình (device, pose_dim...).
        device (torch.device): Device chạy inference.
        compute_dtw_flag (bool): Có tính DTW không (tốn thời gian).
        dtw_max_samples (int): Số mẫu tối đa để tính DTW.

    Returns:
        dict: Dictionary chứa:
            - 'mse': MSE trung bình trên toàn tập.
            - 'dtw': DTW trung bình (nếu compute_dtw_flag=True, else None).
            - 'num_samples': Tổng số mẫu đã evaluate.
    """
    model.eval()

    total_mse = 0.0
    total_samples = 0
    dtw_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):

            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            pose_gt = batch['pose'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            pose_lengths = batch['pose_len']

            # Inference: sample from prior
            outputs = model(
                text_ids=text_ids, text_mask=text_mask,
                target_pose=None, pose_mask=pose_mask,
                teacher_forcing_ratio=0.0
            )
            pose_pred = outputs['pose_pred']

            # MSE (masked - chỉ tính trên valid frames)
            mask_expanded = pose_mask.unsqueeze(-1).float()
            mse_per_element = ((pose_pred - pose_gt) ** 2) * mask_expanded
            batch_mse = mse_per_element.sum() / (mask_expanded.sum() + 1e-8)
            total_mse += batch_mse.item()

            batch_size = text_ids.size(0)
            total_samples += batch_size

            # DTW (tốn thời gian, chỉ tính một phần)
            if compute_dtw_flag and len(dtw_scores) < dtw_max_samples:
                pred_np = pose_pred.cpu().numpy()
                gt_np = pose_gt.cpu().numpy()
                gt_lens = pose_lengths.numpy()
                # pred_len = max_pose_len (không có padding vì inference)
                pred_lens = np.array([config.max_pose_len] * batch_size)

                dtw_val = compute_dtw_batch(pred_np, gt_np, pred_lens, gt_lens)
                dtw_scores.append(dtw_val)

    results = {
        'mse': total_mse / len(dataloader),
        'dtw': float(np.mean(dtw_scores)) if dtw_scores else None,
        'num_samples': total_samples
    }

    return results


def main():
    """
    Entry point: load model và chạy đánh giá đầy đủ trên tập test/dev.

    In kết quả ra console theo dạng bảng IEEE và lưu ra file JSON.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Sign Language CVAE với MSE, DTW, KL metrics."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Đường dẫn đến file checkpoint .pt tốt nhất."
    )
    parser.add_argument(
        "--vocab", type=str, default="data/vocabulary/gloss_vocab.pkl",
        help="Đường dẫn đến file vocabulary .pkl."
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "dev", "test"],
        default="test", help="Tập dữ liệu cần evaluate."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/evaluation",
        help="Thư mục lưu kết quả đánh giá."
    )
    parser.add_argument(
        "--no_dtw", action="store_true",
        help="Bỏ qua tính DTW (nhanh hơn nhưng thiếu metric)."
    )
    parser.add_argument(
        "--dtw_samples", type=int, default=100,
        help="Số mẫu tối đa để tính DTW (DTW có O(n^2) nên giới hạn)."
    )
    args = parser.parse_args()

    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading vocabulary...")
    vocab = GlossVocabulary.load(args.vocab)
    print(f"  Vocab size: {len(vocab)}")

    # Chọn file h5 theo split
    h5_map = {
        "train": config.train_h5,
        "dev": config.dev_h5,
        "test": config.test_h5
    }
    h5_path = h5_map[args.split]

    print(f"Loading {args.split} dataset from {h5_path}...")
    dataset = PhoenixDataset(h5_path, vocab, config.max_text_len, config.max_pose_len)
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )

    print("Loading model from checkpoint...")
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    model = SignLanguageCVAE(
        vocab_size=len(vocab),
        d_model=config.d_model,
        latent_dim=config.latent_dim,
        pose_dim=config.pose_dim,
        max_text_len=config.max_text_len,
        max_pose_len=config.max_pose_len
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded epoch={ckpt.get('epoch', '?')}, val_mse={ckpt.get('val_mse', '?')}")

    # ---- Tính MSE và DTW ----
    print(f"\nEvaluating on {args.split} set ({len(dataset)} samples)...")
    print("  (Tính DTW có thể mất vài phút với data lớn...)")

    results = evaluate(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
        compute_dtw_flag=not args.no_dtw,
        dtw_max_samples=args.dtw_samples
    )

    # ---- Tính KL Latent Space ----
    print("\nComputing latent space KL divergence...")
    latent_kl = compute_latent_kl(model, dataloader, device, max_batches=30)
    results['latent_kl'] = latent_kl
    results['split'] = args.split
    results['checkpoint'] = args.checkpoint

    # ---- In kết quả dạng bảng ----
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Split':<15}: {args.split}")
    print(f"{'Num Samples':<15}: {results['num_samples']}")
    print(f"{'MSE (↓)':<15}: {results['mse']:.6f}")
    if results['dtw'] is not None:
        print(f"{'DTW (↓)':<15}: {results['dtw']:.6f}  (avg over {args.dtw_samples} samples)")
    else:
        print(f"{'DTW':<15}: Skipped (use without --no_dtw to compute)")
    print(f"{'Latent KL (↓)':<15}: {results['latent_kl']:.6f}")
    print("=" * 60)
    print("Note: ↓ = lower is better")

    # ---- Lưu kết quả ra JSON ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"eval_{args.split}_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
