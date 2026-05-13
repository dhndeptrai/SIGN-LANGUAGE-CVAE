# Sign Language CVAE — CSC16004 Final Project

**Text-to-Sign Language Production via Conditional Variational Autoencoder**

> Môn học: CSC16004 - Nhập môn Thị giác Máy tính  
> Nhóm: Đoàn Hải Nam (23122011) & Lê Thái Ngọc (23122012)

---

## Tổng quan kiến trúc

Mô hình chuyển đổi chuỗi Gloss (tiếng Đức) thành chuỗi tọa độ pose người ký hiệu qua 4 thành phần:

```
Text/Gloss → [Transformer Encoder] → context c
                                          |
              [Prior Network p(z|c)] ← c ┘
                       |
                   z ~ N(μ, σ²)
                       |
              [Pose Decoder] ← (z, c) → Pose Sequence [T, 225]
              
Training only:
    [Posterior Encoder q(z|x,c)] ← (pose_gt, c) → z_posterior
```

---

## Cài đặt môi trường

```bash
pip install -r requirements.txt
```

**requirements.txt** bao gồm: `torch`, `h5py`, `numpy`, `mediapipe`, `opencv-python`, `matplotlib`, `scikit-learn`, `tqdm`, `pandas`.

---

## Quy trình chạy toàn bộ pipeline

### Bước 0: Trích xuất Keypoints (M2 đã chạy sẵn)

```bash
# Chạy trên Kaggle — xem file preprocessing/preprocess.py
python preprocessing/preprocess.py
# Output: data/processed/train_data.h5, dev_data.h5, test_data.h5
```

### Bước 1: Build Vocabulary

```bash
python scripts/build_vocabulary.py
# Output: data/vocabulary/gloss_vocab.pkl
#         data/vocabulary/gloss_to_id.json
```

### Bước 2: Training

```bash
# Training đầy đủ (Kaggle/Colab với GPU)
python -m training.train

# Theo dõi progress: checkpoints/training_history.json được cập nhật mỗi epoch
```

**Cấu hình training** (chỉnh trong `training/config.py`):
| Tham số | Giá trị mặc định | Mô tả |
|---------|-----------------|-------|
| `batch_size` | 16 | Giảm xuống 8 nếu OOM |
| `num_epochs` | 50 | Có early stopping |
| `learning_rate` | 1e-4 | Adam optimizer |
| `lambda_mse` | 1.0 | Trọng số MSE loss |
| `lambda_kl` | 0.01 | Trọng số KL (cuối annealing) |
| `lambda_vel` | 0.1 | Trọng số Velocity loss |
| `kl_warmup_epochs` | 10 | Epoch giữ KL=0 trước khi anneal |

### Bước 3: Vẽ Training Curves

```bash
python scripts/plot_training_curves.py \
    --history checkpoints/training_history.json \
    --output_dir outputs/plots/
```

### Bước 4: Đánh giá trên Test Set

```bash
python scripts/evaluate_metrics.py \
    --checkpoint checkpoints/best_model.pt \
    --split test \
    --output_dir outputs/evaluation/

# Nếu muốn bỏ qua DTW (nhanh hơn):
python scripts/evaluate_metrics.py --no_dtw
```

**Output:**
```
EVALUATION RESULTS
============================================================
Split          : test
Num Samples    : 642
MSE (↓)        : 0.012345
DTW (↓)        : 2.345678  (avg over 100 samples)
Latent KL (↓)  : 3.456789
```

### Bước 5: Inference — Sinh pose từ câu Gloss

```bash
python scripts/run_inference.py \
    --checkpoint checkpoints/best_model.pt \
    --gloss "MORGEN SONNE SCHEINEN" \
    --output_dir outputs/poses/ \
    --num_samples 1
# Output: outputs/poses/MORGEN_SONNE_SCHEINEN_sample1.npy
```

### Bước 6: Render Video Người Que (Skeleton Animation)

```bash
python src/visualization/skeleton_renderer.py
# Chỉnh H5_FILE_PATH và OUTPUT_DIRECTORY trong phần __main__ của file
# Output: rendered_videos/*.mp4
```

### Bước 7 (Yêu cầu nâng cao): Khám phá Latent Space

```bash
# Sinh 5 biến thể chuyển động từ cùng 1 câu
python scripts/explore_latent_space.py \
    --mode diversity \
    --checkpoint checkpoints/best_model.pt \
    --gloss "MORGEN SONNE SCHEINEN" \
    --num_samples 5 \
    --output_dir outputs/diversity/

# Visualize latent space bằng t-SNE
python scripts/explore_latent_space.py \
    --mode tsne \
    --checkpoint checkpoints/best_model.pt \
    --output_dir outputs/latent_viz/
```

---

## Cấu trúc thư mục

```
sign-language-cvae/
├── data/
│   ├── processed/          # HDF5 files (train/dev/test_data.h5)
│   └── vocabulary/         # Vocab files (.pkl, .json)
├── checkpoints/            # Model weights, training history
├── outputs/                # Evaluation results, plots, videos
├── src/
│   ├── data/
│   │   ├── dataset.py      # PhoenixDataset (PyTorch Dataset)
│   │   └── vocabulary.py   # GlossVocabulary
│   ├── models/
│   │   ├── cvae.py         # SignLanguageCVAE (main model)
│   │   ├── text_encoder.py # TransformerTextEncoder
│   │   ├── prior_network.py
│   │   ├── posterior_encoder.py
│   │   └── pose_decoder.py # GRU Decoder với Context Injection
│   ├── losses/
│   │   ├── reconstruction.py   # PoseReconstructionLoss (MSE)
│   │   ├── kl_divergence.py    # KLDivergenceLoss
│   │   └── velocity.py         # VelocityLoss + AccelerationLoss
│   └── visualization/
│       └── skeleton_renderer.py
├── training/
│   ├── config.py           # TrainingConfig (tất cả hyperparameters)
│   └── train.py            # Main training loop
└── scripts/
    ├── build_vocabulary.py     # Bước 1: Build vocab
    ├── evaluate_metrics.py     # Bước 4: MSE + DTW + KL
    ├── run_inference.py        # Bước 5: Inference
    ├── explore_latent_space.py # Bước 7: Latent space analysis
    └── plot_training_curves.py # Vẽ đồ thị loss curves
```

---

## Tài liệu tham khảo

1. **Sohn et al. (2015)**: Learning Structured Output Representation using Deep Conditional Generative Models. NeurIPS 2015.
2. **Stoll et al. (2019)**: Text2Sign: Towards Sign Language Production Using Neural Machine Translation and Generative Adversarial Networks. IJCV 2019.
3. **Baltatzis et al. (2024)**: Neural Sign Actors: A diffusion model for 3D sign language production from text. CVPR 2024.
4. **Vaswani et al. (2017)**: Attention Is All You Need. NeurIPS 2017.
5. **Bowman et al. (2016)**: Generating Sentences from a Continuous Space. CoNLL 2016.
6. **Müller (2007)**: Dynamic Time Warping. Springer.

---

*Mọi hàm trong mã nguồn có đầy đủ docstring theo chuẩn Google Style. Xem `CONTEXT` file để biết thêm quy định đồ án.*
