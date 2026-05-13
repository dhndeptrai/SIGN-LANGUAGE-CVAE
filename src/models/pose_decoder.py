"""
Pose Decoder: Giải mã latent z và context c thành chuỗi pose.

Cải tiến so với phiên bản gốc:
    1. Learned Start Token: Thay zero-vector bằng một vector có thể học
       được (nn.Parameter), giải quyết cold-start problem ở frame đầu tiên.
    2. Context Injection mỗi step: Thay vì chỉ dùng z+c để khởi tạo
       hidden state, context được nối vào input của GRU mỗi bước thời gian.
       Điều này giúp decoder "nhớ" nội dung text trong suốt quá trình sinh.
    3. Layer Normalization: Thêm LayerNorm sau output projection để ổn định
       giá trị tọa độ đầu ra.

Tham khảo:
    - Sohn et al. (2015): CVAE gốc.
    - Stoll et al. (2019): Text2Sign decoder architecture.
    - Graves (2013): Generating sequences with RNNs - scheduled sampling.
      https://arxiv.org/abs/1308.0850
"""

import torch
import torch.nn as nn
import random


class PoseDecoder(nn.Module):
    """
    Autoregressive GRU Decoder sinh chuỗi pose từ latent z và context c.

    Architecture (nâng cấp):
        [z ; c] -> fc_init -> h_0 (initial hidden state)
        Mỗi bước t:
            input_t = [prev_pose ; c]   <- context injection mỗi step
            gru_out, h_t = GRU(input_t, h_{t-1})
            pose_t = LayerNorm(fc_out(gru_out))

    Learned start token:
        prev_pose tại t=0 là nn.Parameter (learnable), không phải zero-vector.
        Điều này giải quyết hiện tượng frame đầu tiên bị méo.

    Attributes:
        pose_dim (int): Dimension của 1 frame pose (225 cho MediaPipe Holistic).
        hidden_dim (int): Hidden dimension của GRU.
        max_len (int): Độ dài tối đa chuỗi sinh ra khi không có target_pose.
        start_token (nn.Parameter): Learned start token, shape [1, 1, pose_dim].
    """

    def __init__(
        self,
        latent_dim: int = 128,
        context_dim: int = 512,
        pose_dim: int = 225,
        hidden_dim: int = 512,
        num_layers: int = 2,
        max_len: int = 300,
        dropout: float = 0.1
    ):
        """
        Args:
            latent_dim (int): Dimension của latent z.
            context_dim (int): Dimension của context vector c từ Text Encoder.
            pose_dim (int): Dimension của output pose mỗi frame.
            hidden_dim (int): Hidden dimension của GRU.
            num_layers (int): Số lớp GRU.
            max_len (int): Độ dài tối đa khi inference (không có target_pose).
            dropout (float): Dropout rate giữa các GRU layer.
        """
        super().__init__()

        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.context_dim = context_dim

        # FC khởi tạo hidden state từ [z ; c]
        self.fc_init = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim * num_layers),
            nn.Tanh()   # tanh để giữ hidden state trong range hợp lý
        )

        # Learned start token: thay thế zero-vector, fix cold-start problem
        # Shape: [1, 1, pose_dim] để broadcast theo batch
        self.start_token = nn.Parameter(
            torch.zeros(1, 1, pose_dim)
        )
        nn.init.normal_(self.start_token, mean=0.0, std=0.01)

        # GRU decoder: input là [prev_pose ; context] -> pose_dim + context_dim
        self.gru = nn.GRU(
            input_size=pose_dim + context_dim,   # context injection mỗi step
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection với LayerNorm để ổn định giá trị tọa độ
        self.fc_out = nn.Linear(hidden_dim, pose_dim)
        self.layer_norm = nn.LayerNorm(pose_dim)

    def _init_hidden(
        self,
        z: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Tạo hidden state ban đầu từ latent z và context c.

        Args:
            z (torch.Tensor): Shape [batch, latent_dim].
            context (torch.Tensor): Shape [batch, context_dim].

        Returns:
            torch.Tensor: Shape [num_layers, batch, hidden_dim].
        """
        combined = torch.cat([z, context], dim=-1)   # [batch, latent+context]
        h0 = self.fc_init(combined)                  # [batch, hidden*num_layers]
        # Reshape thành [num_layers, batch, hidden_dim]
        h0 = h0.view(-1, self.num_layers, self.hidden_dim)
        h0 = h0.transpose(0, 1).contiguous()
        return h0

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        target_pose: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Decode latent z thành chuỗi pose autoregressive.

        Training mode (target_pose != None):
            Với xác suất teacher_forcing_ratio, dùng ground truth frame t-1
            làm input cho bước t. Ngược lại dùng prediction.

        Inference mode (target_pose = None):
            Luôn dùng prediction của bước trước làm input bước sau.
            Bước đầu tiên dùng learned start_token.

        Args:
            z (torch.Tensor): Latent vector, shape [batch, latent_dim].
            context (torch.Tensor): Context vector, shape [batch, context_dim].
            target_pose (torch.Tensor): Ground truth pose [batch, seq_len, pose_dim].
                                        None khi inference.
            teacher_forcing_ratio (float): Xác suất dùng ground truth làm input.
                                           Nên giảm dần theo epoch.

        Returns:
            torch.Tensor: Predicted pose sequence, shape [batch, seq_len, pose_dim].
        """
        batch_size = z.size(0)
        seq_len = target_pose.size(1) if target_pose is not None else self.max_len

        # Khởi tạo hidden state từ [z ; c]
        hidden = self._init_hidden(z, context)  # [num_layers, batch, hidden_dim]

        # Learned start token, broadcast theo batch
        # [1, 1, pose_dim] -> [batch, 1, pose_dim]
        decoder_input_pose = self.start_token.expand(batch_size, 1, self.pose_dim)

        # Context được lặp lại mỗi bước: [batch, 1, context_dim]
        context_step = context.unsqueeze(1)  # [batch, 1, context_dim]

        outputs = []

        for t in range(seq_len):
            # Nối pose input với context: [batch, 1, pose_dim + context_dim]
            gru_input = torch.cat(
                [decoder_input_pose, context_step], dim=-1
            )

            # GRU step
            gru_out, hidden = self.gru(gru_input, hidden)
            # gru_out: [batch, 1, hidden_dim]

            # Project -> pose với LayerNorm
            pose_pred = self.layer_norm(
                self.fc_out(gru_out.squeeze(1))
            )  # [batch, pose_dim]

            outputs.append(pose_pred.unsqueeze(1))  # [batch, 1, pose_dim]

            # Quyết định input cho bước tiếp theo
            if target_pose is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing: dùng ground truth
                decoder_input_pose = target_pose[:, t:t+1, :]
            else:
                # Scheduled sampling: dùng prediction
                decoder_input_pose = pose_pred.unsqueeze(1).detach()

        # Stack toàn bộ outputs
        pose_sequence = torch.cat(outputs, dim=1)  # [batch, seq_len, pose_dim]
        return pose_sequence
