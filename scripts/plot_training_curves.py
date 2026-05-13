"""
Vẽ đồ thị training curves từ file training_history.json.

Dùng để báo cáo kết quả thực nghiệm (Tuần 3):
    - Loss curves (Train/Val MSE theo epoch).
    - KL Divergence curve theo epoch.
    - Velocity Loss curve theo epoch.
    - KL Weight annealing schedule.

Cách dùng:
    python scripts/plot_training_curves.py \
        --history checkpoints/training_history.json \
        --output_dir outputs/plots/
"""

import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_training_history(history_path: str) -> list:
    """
    Đọc file training_history.json đã lưu từ training script.

    Args:
        history_path (str): Đường dẫn đến file JSON.

    Returns:
        list[dict]: Danh sách metrics theo từng epoch.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, 'r') as f:
        history = json.load(f)

    print(f"  Loaded history: {len(history)} epochs")
    return history


def plot_loss_curves(history: list, output_dir: str) -> None:
    """
    Vẽ 4 biểu đồ trên cùng 1 figure:
        (1) Total Train Loss
        (2) Train MSE vs Val MSE
        (3) Train KL Divergence
        (4) Train Velocity Loss + KL Weight Schedule

    Args:
        history (list[dict]): Danh sách metrics theo epoch từ training.
        output_dir (str): Thư mục lưu ảnh output.

    Returns:
        None. Ảnh được lưu tại output_dir/training_curves.png.
    """
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['loss'] for h in history]
    train_mse = [h['train']['mse'] for h in history]
    train_kl = [h['train']['kl'] for h in history]
    train_vel = [h['train']['vel'] for h in history]
    val_mse = [h['val']['mse'] for h in history]
    kl_weights = [h.get('kl_weight', 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves - Sign Language CVAE", fontsize=16, fontweight='bold')

    # (1) Total Train Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', linewidth=2, label='Total Train Loss')
    axes[0, 0].set_title("Total Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (2) MSE: Train vs Val
    axes[0, 1].plot(epochs, train_mse, 'b-', linewidth=2, label='Train MSE')
    axes[0, 1].plot(epochs, val_mse, 'r--', linewidth=2, label='Val MSE')
    best_epoch = epochs[int(np.argmin(val_mse))]
    best_val = min(val_mse)
    axes[0, 1].axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7,
                       label=f'Best Val @ ep{best_epoch} = {best_val:.4f}')
    axes[0, 1].set_title("Reconstruction Loss (MSE)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (3) KL Divergence
    axes[1, 0].plot(epochs, train_kl, 'purple', linewidth=2, label='KL Divergence')
    ax_kl2 = axes[1, 0].twinx()
    ax_kl2.plot(epochs, kl_weights, 'orange', linestyle='--', linewidth=1.5,
                alpha=0.7, label='KL Weight (annealed)')
    ax_kl2.set_ylabel("KL Weight", color='orange')
    axes[1, 0].set_title("KL Divergence + Annealing Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("KL Divergence")
    axes[1, 0].legend(loc='upper left')
    ax_kl2.legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    # (4) Velocity Loss
    axes[1, 1].plot(epochs, train_vel, 'green', linewidth=2, label='Velocity Loss')
    axes[1, 1].set_title("Velocity Smoothness Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Velocity Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    print(f"  Best Val MSE: {best_val:.4f} at epoch {best_epoch}")


def print_summary_table(history: list) -> None:
    """
    In bảng tóm tắt kết quả thực nghiệm ra console.

    In 5 epoch đầu, 5 epoch cuối, và epoch có val_mse tốt nhất.
    Dùng để copy vào báo cáo IEEE.

    Args:
        history (list[dict]): Danh sách metrics theo epoch.

    Returns:
        None.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train MSE':>9} | "
          f"{'Train KL':>8} | {'Train Vel':>9} | {'Val MSE':>7}")
    print("-" * 80)

    best_epoch_idx = int(np.argmin([h['val']['mse'] for h in history]))
    highlight_epochs = set(
        [h['epoch'] for h in history[:5]]
        + [h['epoch'] for h in history[-5:]]
        + [history[best_epoch_idx]['epoch']]
    )

    for h in history:
        marker = " ★" if h['epoch'] == history[best_epoch_idx]['epoch'] else ""
        if h['epoch'] in highlight_epochs:
            print(
                f"{h['epoch']:>6} | "
                f"{h['train']['loss']:>10.4f} | "
                f"{h['train']['mse']:>9.4f} | "
                f"{h['train']['kl']:>8.4f} | "
                f"{h['train']['vel']:>9.4f} | "
                f"{h['val']['mse']:>7.4f}{marker}"
            )
    print("=" * 80)
    best = history[best_epoch_idx]
    print(f"Best epoch: {best['epoch']} | Val MSE: {best['val']['mse']:.4f} "
          f"| Val Vel: {best['val']['vel']:.4f}")


def main():
    """
    Entry point: load history và vẽ đồ thị.
    """
    parser = argparse.ArgumentParser(
        description="Vẽ training curves từ training_history.json"
    )
    parser.add_argument(
        "--history",
        type=str,
        default="checkpoints/training_history.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/plots"
    )
    args = parser.parse_args()

    print("Loading training history...")
    history = load_training_history(args.history)

    print("Plotting curves...")
    plot_loss_curves(history, args.output_dir)

    print_summary_table(history)
    print("\nDone!")


if __name__ == "__main__":
    main()
