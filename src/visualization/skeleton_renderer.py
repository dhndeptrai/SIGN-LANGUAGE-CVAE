"""
Skeleton Renderer: Vẽ chuỗi pose keypoints thành video người que (stickman).

Hỗ trợ 2 nguồn dữ liệu:
    - File HDF5 (ground truth từ MediaPipe, dùng cho evaluation).
    - File .npy (output của CVAE model, dùng cho demo).

Bug fixes so với phiên bản gốc:
    1. Thêm đầy đủ connections chân (25-32) và nose->shoulder (cổ/đầu).
    2. Sửa palm connections còn thiếu: wrist->middle_mcp và wrist->ring_mcp.
    3. Normalize tọa độ đúng cách: per-part (body/left_hand/right_hand riêng)
       thay vì global range, tránh tay bị co cụm vào giữa thân.
    4. Thêm y-flip cho tọa độ normalized [0,1] của MediaPipe
       (MediaPipe y=0 ở trên, OpenCV y=0 ở trên -> không cần flip,
        nhưng cần kiểm tra range thực tế và center vào canvas).

Tham khảo:
    - MediaPipe Holistic landmark indices:
      https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker
    - Baltatzis et al. (2024): Neural Sign Actors - skeleton visualization.
"""

import os
import cv2
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class SkeletonRenderer:
    """
    Render chuỗi pose keypoints thành video skeleton animation.

    Layout tọa độ đầu vào (MediaPipe Holistic, 75 landmarks):
        - Index   0 - 32 : Body pose (33 points)
        - Index  33 - 53 : Left hand (21 points)
        - Index  54 - 74 : Right hand (21 points)

    Attributes:
        h5_filepath (Path): Đường dẫn file HDF5 (dùng khi render từ ground truth).
        output_dir (Path): Thư mục lưu video đầu ra.
        dataset_name (str): Tên dataset bên trong HDF5 group.
    """

    # ------------------------------------------------------------------
    # MediaPipe Pose body connections (0-32)
    # Thêm đầy đủ: mặt, vai, tay, hông, chân
    # Thêm (0,11) và (0,12) để vẽ cổ/đầu -- bug gốc thiếu đây
    # ------------------------------------------------------------------
    BODY_CONNECTIONS: List[Tuple[int, int]] = [
        # Mặt
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        # Cổ / đầu nối xuống vai -- FIX: thêm 2 connections này
        (0, 11), (0, 12),
        # Vai và thân trên
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        # Bàn tay (fingertip connections với wrist đã có qua tay riêng)
        (15, 17), (15, 19), (15, 21), (17, 19),
        (16, 18), (16, 20), (16, 22), (18, 20),
        # Hông
        (11, 23), (12, 24), (23, 24),
        # Chân -- FIX: thêm toàn bộ leg connections (25-32 bị thiếu hoàn toàn)
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    ]

    # ------------------------------------------------------------------
    # Left hand connections (offset 33, MediaPipe hand 0-20)
    # FIX: thêm wrist->middle_mcp (33,42) và wrist->ring_mcp (33,46)
    # để lòng bàn tay không bị hở
    # ------------------------------------------------------------------
    LEFT_HAND_CONNECTIONS: List[Tuple[int, int]] = [
        # Ngón cái: wrist->cmc->mcp->ip->tip
        (33, 34), (34, 35), (35, 36), (36, 37),
        # Ngón trỏ: wrist->mcp->pip->dip->tip
        (33, 38), (38, 39), (39, 40), (40, 41),
        # Ngón giữa: wrist->mcp->pip->dip->tip
        (33, 42), (42, 43), (43, 44), (44, 45),   # FIX: thêm (33,42)
        # Ngón áp út: wrist->mcp->pip->dip->tip
        (33, 46), (46, 47), (47, 48), (48, 49),   # FIX: thêm (33,46)
        # Ngón út: wrist->mcp->pip->dip->tip
        (33, 50), (50, 51), (51, 52), (52, 53),
        # Lòng bàn tay (nối các MCP với nhau)
        (38, 42), (42, 46), (46, 50),
    ]

    # ------------------------------------------------------------------
    # Right hand connections (offset 54, cấu trúc tương tự left hand)
    # FIX: thêm (54,63) và (54,67) tương tự
    # ------------------------------------------------------------------
    RIGHT_HAND_CONNECTIONS: List[Tuple[int, int]] = [
        (54, 55), (55, 56), (56, 57), (57, 58),
        (54, 59), (59, 60), (60, 61), (61, 62),
        (54, 63), (63, 64), (64, 65), (65, 66),   # FIX: thêm (54,63)
        (54, 67), (67, 68), (68, 69), (69, 70),   # FIX: thêm (54,67)
        (54, 71), (71, 72), (72, 73), (73, 74),
        (59, 63), (63, 67), (67, 71),
    ]

    CONNECTIONS: List[Tuple[int, int]] = (
        BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS
    )

    # Màu sắc theo từng phần cơ thể
    COLOR_BODY = (200, 200, 200)       # xám sáng
    COLOR_LEFT_HAND = (80, 200, 120)   # xanh lá
    COLOR_RIGHT_HAND = (80, 120, 220)  # xanh dương
    COLOR_JOINT = (0, 165, 255)        # cam

    def __init__(
        self,
        h5_filepath: str = "",
        output_dir: str = "rendered_videos",
        dataset_name: str = "keypoints"
    ):
        """
        Khởi tạo SkeletonRenderer.

        Args:
            h5_filepath (str): Đường dẫn file HDF5. Có thể để trống nếu
                               chỉ dùng render_from_npy().
            output_dir (str): Thư mục lưu video đầu ra.
            dataset_name (str): Tên dataset bên trong HDF5 group.
        """
        self.h5_filepath = Path(h5_filepath) if h5_filepath else None
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal: normalize tọa độ
    # ------------------------------------------------------------------

    def _normalize_landmarks_to_canvas(
        self,
        landmarks: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        padding: float = 0.1
    ) -> np.ndarray:
        """
        Chuẩn hóa tọa độ [x, y] của toàn bộ 75 landmarks vào canvas pixel.

        Chiến lược: GLOBAL normalization — dùng range của toàn bộ skeleton
        (không phải per-part) để tính scale và offset.

        Lý do dùng global thay vì per-part:
            Per-part normalize phá vỡ spatial relationship giữa body và tay.
            Ví dụ: body ở (0.3, 0.5), tay ở (0.28, 0.48) — rất gần nhau.
            Per-part sẽ kéo mỗi phần ra giữa canvas riêng -> chúng chồng nhau.
            Global giữ đúng vị trí tương đối: tay luôn gần vai.

        Xử lý tọa độ model output (không phải MediaPipe normalized):
            Model CVAE sinh tọa độ raw (thường centered tại 0, range tùy).
            KHÔNG dùng heuristic "max_val <= 2.0" vì dễ sai.
            LUÔN dùng min-max global normalize để an toàn.

        Args:
            landmarks (np.ndarray): Shape [75, 3] hoặc [75, 2].
            canvas_w (int): Chiều rộng canvas pixel.
            canvas_h (int): Chiều cao canvas pixel.
            padding (float): Padding xung quanh skeleton (0.1 = 10% mỗi phía).

        Returns:
            np.ndarray: Shape [75, 2]. Tọa độ pixel trên canvas.
                        NaN cho landmarks bị missing/invalid.
        """
        xy = landmarks[:, :2].copy().astype(np.float32)  # [75, 2]

        # Mask landmarks hợp lệ: không phải NaN, không phải (0,0) hoàn toàn
        valid_mask = ~(np.isnan(xy).any(axis=1) | ((xy[:, 0] == 0) & (xy[:, 1] == 0)))

        if valid_mask.sum() < 2:
            return np.full_like(xy, np.nan)

        valid_xy = xy[valid_mask]

        # --- Global min-max normalize, giữ aspect ratio ---
        x_min, x_max = valid_xy[:, 0].min(), valid_xy[:, 0].max()
        y_min, y_max = valid_xy[:, 1].min(), valid_xy[:, 1].max()

        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        # Vùng canvas có thể vẽ (bỏ padding)
        pad_px = padding * canvas_w
        pad_py = padding * canvas_h
        usable_w = canvas_w - 2 * pad_px
        usable_h = canvas_h - 2 * pad_py

        # Scale đồng nhất (không méo): dùng min của 2 chiều
        scale = min(usable_w / x_range, usable_h / y_range)

        # Center của skeleton trong tọa độ gốc
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        # Áp dụng transform: translate về 0, scale, translate ra giữa canvas
        pixel_xy = np.full_like(xy, np.nan)
        pixel_xy[valid_mask, 0] = (valid_xy[:, 0] - cx) * scale + canvas_w / 2.0
        pixel_xy[valid_mask, 1] = (valid_xy[:, 1] - cy) * scale + canvas_h / 2.0

        return pixel_xy

    def _get_connection_color(self, idx1: int, idx2: int) -> Tuple[int, int, int]:
        """
        Trả về màu của đường nối dựa theo phần cơ thể.

        Args:
            idx1 (int): Index landmark đầu tiên.
            idx2 (int): Index landmark thứ hai.

        Returns:
            Tuple[int, int, int]: Màu BGR cho OpenCV.
        """
        if idx1 >= 54 or idx2 >= 54:
            return self.COLOR_RIGHT_HAND
        if idx1 >= 33 or idx2 >= 33:
            return self.COLOR_LEFT_HAND
        return self.COLOR_BODY

    def _draw_single_frame(
        self,
        frame_landmarks: np.ndarray,
        frame_size: Tuple[int, int] = (800, 800)
    ) -> np.ndarray:
        """
        Vẽ một frame skeleton từ mảng tọa độ 75 landmarks.

        Args:
            frame_landmarks (np.ndarray): Shape [75, 3] hoặc [75, 2].
                                          Tọa độ của 75 keypoints trong 1 frame.
            frame_size (Tuple[int, int]): (width, height) của canvas pixel.

        Returns:
            np.ndarray: Ảnh BGR shape [height, width, 3].
        """
        canvas_w, canvas_h = frame_size
        frame = np.full((canvas_h, canvas_w, 3), (40, 40, 40), dtype=np.uint8)

        num_landmarks = frame_landmarks.shape[0]
        if num_landmarks < 75:
            # Pad NaN nếu thiếu landmarks
            padded = np.full((75, frame_landmarks.shape[1]), np.nan)
            padded[:num_landmarks] = frame_landmarks
            frame_landmarks = padded

        # Chuẩn hóa tọa độ ra pixel
        pixel_xy = self._normalize_landmarks_to_canvas(
            frame_landmarks, canvas_w, canvas_h, padding=0.05
        )

        # Vẽ xương (connections)
        for conn in self.CONNECTIONS:
            idx1, idx2 = conn
            if idx1 >= 75 or idx2 >= 75:
                continue

            x1, y1 = pixel_xy[idx1]
            x2, y2 = pixel_xy[idx2]

            # Bỏ qua nếu NaN hoặc out-of-canvas
            if (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                continue
            if not (0 <= x1 < canvas_w and 0 <= y1 < canvas_h):
                continue
            if not (0 <= x2 < canvas_w and 0 <= y2 < canvas_h):
                continue

            color = self._get_connection_color(idx1, idx2)
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.line(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

        # Vẽ khớp (joints)
        for i in range(75):
            x, y = pixel_xy[i]
            if np.isnan(x) or np.isnan(y):
                continue
            if not (0 <= x < canvas_w and 0 <= y < canvas_h):
                continue
            cv2.circle(
                frame, (int(x), int(y)),
                radius=4, color=self.COLOR_JOINT,
                thickness=-1, lineType=cv2.LINE_AA
            )

        return frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_from_array(
        self,
        video_name: str,
        landmarks_sequence: np.ndarray,
        fps: int = 25,
        frame_size: Tuple[int, int] = (800, 800)
    ) -> str:
        """
        Render chuỗi pose và lưu thành file .mp4.

        Chấp nhận 2 shapes:
            - [T, 225]: flat array (output của CVAE model) -> auto reshape sang [T, 75, 3].
            - [T, 75, 3]: đã reshape sẵn (output từ MediaPipe).

        Args:
            video_name (str): Tên file output (không cần extension).
            landmarks_sequence (np.ndarray): Chuỗi pose với shape [T, 225]
                                             hoặc [T, 75, 3].
            fps (int): Frames per second của video output.
            frame_size (Tuple[int, int]): (width, height) canvas.

        Returns:
            str: Đường dẫn tuyệt đối đến file .mp4 đã tạo.

        Raises:
            ValueError: Nếu shape không hợp lệ.
        """
        # Reshape nếu cần
        if landmarks_sequence.ndim == 2:
            num_frames, total_features = landmarks_sequence.shape
            if total_features % 3 != 0:
                raise ValueError(
                    f"Shape {landmarks_sequence.shape} không hợp lệ: "
                    f"total_features={total_features} không chia hết cho 3."
                )
            num_landmarks = total_features // 3
            landmarks_sequence = landmarks_sequence.reshape(num_frames, num_landmarks, 3)
        elif landmarks_sequence.ndim == 3:
            num_frames = landmarks_sequence.shape[0]
        else:
            raise ValueError(f"landmarks_sequence phải có 2 hoặc 3 chiều, nhận {landmarks_sequence.ndim}.")

        output_path = self.output_dir / f"{video_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

        for t in range(num_frames):
            rendered = self._draw_single_frame(landmarks_sequence[t], frame_size)
            out.write(rendered)

        out.release()
        logging.info(f"Video saved: {output_path}")
        return str(output_path)

    def render_from_npy(
        self,
        npy_path: str,
        fps: int = 25,
        frame_size: Tuple[int, int] = (800, 800)
    ) -> str:
        """
        Shortcut: load file .npy rồi render thành video.

        Dùng cho output của CVAE inference script.

        Args:
            npy_path (str): Đường dẫn đến file .npy, shape [T, 225].
            fps (int): Frames per second.
            frame_size (Tuple[int, int]): (width, height) canvas.

        Returns:
            str: Đường dẫn đến file .mp4 đã tạo.
        """
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"File not found: {npy_path}")

        data = np.load(str(npy_path))  # [T, 225]
        video_name = npy_path.stem     # tên file không có extension

        logging.info(f"Loaded .npy: shape={data.shape}, rendering '{video_name}'...")
        return self.render_from_array(video_name, data, fps, frame_size)

    def load_h5_data(self) -> Dict[str, np.ndarray]:
        """
        Đọc dữ liệu landmark từ file HDF5.

        Cấu trúc HDF5 mong đợi:
            file.h5
            └── <video_name> (Group)
                └── keypoints (Dataset, shape [T, 75, 3] hoặc [T, 225])

        Returns:
            Dict[str, np.ndarray]: Mapping từ tên video sang mảng tọa độ.

        Raises:
            FileNotFoundError: Nếu file HDF5 không tồn tại.
            KeyError: Nếu lỗi parse bên trong HDF5.
        """
        if self.h5_filepath is None or not self.h5_filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_filepath}")

        data_dict = {}
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                for group_name in f.keys():
                    group = f[group_name]
                    if self.dataset_name in group:
                        data_dict[group_name] = np.array(group[self.dataset_name])
                    else:
                        logging.warning(f"Skip '{group_name}': dataset '{self.dataset_name}' not found.")
        except Exception as e:
            raise KeyError(f"Error parsing HDF5: {e}")

        logging.info(f"Loaded {len(data_dict)} videos from HDF5.")
        return data_dict

    def render_all_from_h5(self, fps: int = 25, frame_size: Tuple[int, int] = (800, 800)) -> None:
        """
        Render toàn bộ video trong file HDF5.

        Args:
            fps (int): Frames per second.
            frame_size (Tuple[int, int]): (width, height) canvas.
        """
        all_data = self.load_h5_data()
        logging.info(f"Rendering {len(all_data)} videos...")
        for video_name, seq_data in all_data.items():
            self.render_from_array(video_name, seq_data, fps, frame_size)
        logging.info("Done rendering all videos.")


# ==========================================
# MAIN: Demo render từ .npy
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render pose .npy thành video skeleton.")
    parser.add_argument("--npy", type=str, required=True,
                        help="Đường dẫn file .npy (shape [T, 225])")
    parser.add_argument("--output_dir", type=str, default="rendered_videos")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    renderer = SkeletonRenderer(output_dir=args.output_dir)
    out_path = renderer.render_from_npy(args.npy, fps=args.fps)
    print(f"Output: {out_path}")
