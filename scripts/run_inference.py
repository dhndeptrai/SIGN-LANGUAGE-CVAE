"""
Script inference: Sinh pose sequence từ câu gloss đầu vào.

Chức năng:
    - Load model từ checkpoint.
    - Nhận câu gloss (tiếng Đức) làm input.
    - Sinh pose sequence và lưu ra file .npy để M2 dùng Skeleton Renderer.

Cách dùng:
    python scripts/run_inference.py \
        --checkpoint checkpoints/best_model.pt \
        --gloss "MORGEN SONNE SCHEINEN" \
        --output_dir outputs/

Tham khảo:
    - Stoll et al. (2019): Text2Sign inference pipeline.
"""

import argparse
import os
import numpy as np
import torch

from src.data.vocabulary import GlossVocabulary
from src.models.cvae import SignLanguageCVAE
from training.config import TrainingConfig


def load_model_from_checkpoint(
    checkpoint_path: str,
    vocab_size: int,
    config: TrainingConfig
) -> SignLanguageCVAE:
    """
    Load mô hình CVAE từ file checkpoint đã lưu.

    Args:
        checkpoint_path (str): Đường dẫn đến file .pt checkpoint.
        vocab_size (int): Kích thước vocabulary (dùng để khởi tạo model).
        config (TrainingConfig): Cấu hình kiến trúc model.

    Returns:
        SignLanguageCVAE: Mô hình đã load weights, ở eval mode.

    Raises:
        FileNotFoundError: Nếu checkpoint_path không tồn tại.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = SignLanguageCVAE(
        vocab_size=vocab_size,
        d_model=config.d_model,
        latent_dim=config.latent_dim,
        pose_dim=config.pose_dim,
        max_text_len=config.max_text_len,
        max_pose_len=config.max_pose_len
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    val_mse = checkpoint.get('val_mse', 'unknown')
    print(f"  Loaded checkpoint: epoch={epoch}, val_mse={val_mse}")

    return model


def tokenize_gloss(
    gloss_sentence: str,
    vocab: GlossVocabulary,
    max_len: int
) -> tuple:
    """
    Chuyển đổi câu gloss thành tensor token IDs và attention mask.

    Args:
        gloss_sentence (str): Câu gloss, các gloss cách nhau bởi dấu cách.
                              Ví dụ: "MORGEN SONNE SCHEINEN".
        vocab (GlossVocabulary): Vocabulary đã build từ tập train.
        max_len (int): Độ dài tối đa của sequence (padding/truncation).

    Returns:
        tuple:
            - token_ids (torch.Tensor): Shape [1, max_len], dtype=long.
            - attention_mask (torch.Tensor): Shape [1, max_len], dtype=bool.
              True = token thật, False = padding.
    """
    tokens = gloss_sentence.strip().upper().split()
    token_ids = [vocab.get_id(token) for token in tokens]

    # Truncate nếu dài quá
    token_ids = token_ids[:max_len]
    seq_len = len(token_ids)

    # Padding
    padded = token_ids + [vocab.pad_id] * (max_len - seq_len)

    token_tensor = torch.tensor([padded], dtype=torch.long)
    mask = torch.zeros(1, max_len, dtype=torch.bool)
    mask[0, :seq_len] = True

    return token_tensor, mask


def run_inference(
    model: SignLanguageCVAE,
    gloss_sentence: str,
    vocab: GlossVocabulary,
    config: TrainingConfig,
    device: torch.device,
    num_samples: int = 1
) -> np.ndarray:
    """
    Sinh pose sequences từ một câu gloss.

    Args:
        model (SignLanguageCVAE): Mô hình đã load.
        gloss_sentence (str): Câu gloss đầu vào.
        vocab (GlossVocabulary): Vocabulary.
        config (TrainingConfig): Cấu hình model.
        device (torch.device): Device chạy inference.
        num_samples (int): Số lượng biến thể pose cần sinh ra.

    Returns:
        np.ndarray: Shape [num_samples, seq_len, pose_dim].
                    Mảng tọa độ pose đã sinh.
    """
    token_ids, attention_mask = tokenize_gloss(
        gloss_sentence, vocab, config.max_text_len
    )
    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        pose_samples = model.generate(
            text_ids=token_ids,
            text_mask=attention_mask,
            num_samples=num_samples
        )

    return pose_samples.cpu().numpy()


def main():
    """
    Entry point: parse arguments và chạy inference.
    """
    parser = argparse.ArgumentParser(
        description="Run inference on Sign Language CVAE model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Đường dẫn đến file checkpoint .pt"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/vocabulary/gloss_vocab.pkl",
        help="Đường dẫn đến file vocabulary .pkl"
    )
    parser.add_argument(
        "--gloss",
        type=str,
        required=True,
        help="Câu gloss tiếng Đức, ví dụ: 'MORGEN SONNE SCHEINEN'"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Thư mục lưu file .npy kết quả"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Số lượng biến thể pose cần sinh (dùng cho yêu cầu nâng cao)"
    )
    args = parser.parse_args()

    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading vocabulary...")
    vocab = GlossVocabulary.load(args.vocab)
    print(f"  Vocab size: {len(vocab)}")

    print("Loading model...")
    model = load_model_from_checkpoint(args.checkpoint, len(vocab), config)
    model = model.to(device)

    print(f"Running inference for: '{args.gloss}'")
    pose_samples = run_inference(
        model=model,
        gloss_sentence=args.gloss,
        vocab=vocab,
        config=config,
        device=device,
        num_samples=args.num_samples
    )

    print(f"Generated {args.num_samples} sample(s), shape: {pose_samples.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    safe_gloss = args.gloss.replace(" ", "_")[:30]

    for i, sample in enumerate(pose_samples):
        output_path = os.path.join(args.output_dir, f"{safe_gloss}_sample{i+1}.npy")
        np.save(output_path, sample)
        print(f"  Saved: {output_path} (shape={sample.shape})")

    print("\nDone! M2 có thể dùng file .npy này với Skeleton Renderer.")


if __name__ == "__main__":
    main()
