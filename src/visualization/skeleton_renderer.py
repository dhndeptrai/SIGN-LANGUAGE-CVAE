import os
import cv2
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Cấu hình logging cơ bản để tracking quá trình xử lý
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SkeletonRenderer:
    """
    Lớp hỗ trợ đọc dữ liệu landmark từ file HDF5 và xuất ra video người que (stickman).
    
    Attributes:
        h5_filepath (Path): Đường dẫn đến file .h5 chứa dữ liệu.
        output_dir (Path): Thư mục lưu các video đầu ra.
        dataset_name (str): Tên của dataset chứa mảng tọa độ trong mỗi Group.
    """
    
    # 33 điểm Pose + 21 điểm Tay Trái (33->53) + 21 điểm Tay Phải (54->74)
    CONNECTIONS: List[Tuple[int, int]] = [
        # Thân và mặt (MediaPipe Pose 0-32)
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        
        # Bàn tay trái (Từ index 33 đến 53)
        (33, 34), (34, 35), (35, 36), (36, 37), # Ngón cái
        (33, 38), (38, 39), (39, 40), (40, 41), # Ngón trỏ
        (38, 42), (42, 43), (43, 44), (44, 45), # Ngón giữa
        (42, 46), (46, 47), (47, 48), (48, 49), # Ngón áp út
        (46, 50), (50, 51), (51, 52), (52, 53), # Ngón út
        (33, 50),                               # Nối lòng bàn tay

        # Bàn tay phải (Từ index 54 đến 74)
        (54, 55), (55, 56), (56, 57), (57, 58), # Ngón cái
        (54, 59), (59, 60), (60, 61), (61, 62), # Ngón trỏ
        (59, 63), (63, 64), (64, 65), (65, 66), # Ngón giữa
        (63, 67), (67, 68), (68, 69), (69, 70), # Ngón áp út
        (67, 71), (71, 72), (72, 73), (73, 74), # Ngón út
        (54, 71)                                # Nối lòng bàn tay
    ]

    def __init__(self, h5_filepath: str, output_dir: str, dataset_name: str = 'keypoints'):
        self.h5_filepath = Path(h5_filepath)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        
        # Tự động tạo thư mục output nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_h5_data(self) -> Dict[str, np.ndarray]:
        """
        Đọc dữ liệu từ file HDF5 theo cấu trúc: Group (tên video) -> Dataset (tọa độ).

        Returns:
            Dict[str, np.ndarray]: Dictionary ánh xạ từ tên video sang mảng tọa độ numpy.
            
        Raises:
            FileNotFoundError: Nếu không tìm thấy file .h5.
            KeyError: Nếu xảy ra lỗi truy xuất dataset bên trong file.
        """
        data_dict = {}
        
        if not self.h5_filepath.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {self.h5_filepath}")

        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                for group_name in f.keys():
                    group = f[group_name]
                    
                    # Kiểm tra Dataset có tồn tại trong Group hay không
                    if self.dataset_name in group:
                        # Load toàn bộ mảng numpy vào memory
                        data_dict[group_name] = np.array(group[self.dataset_name])
                    else:
                        logging.warning(f"Bỏ qua '{group_name}': Không tìm thấy dataset '{self.dataset_name}'.")
                        
        except Exception as e:
            raise KeyError(f"Đã xảy ra lỗi khi parse file HDF5: {str(e)}")
            
        logging.info(f"Đã tải thành công dữ liệu của {len(data_dict)} video.")
        return data_dict

    def _draw_single_frame(self, frame_landmarks: np.ndarray, frame_size: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Vẽ một khung hình người que từ mảng tọa độ của 1 frame.
        Hỗ trợ tự động scale nếu tọa độ bị chuẩn hóa [0, 1].
        """
        frame = np.full((frame_size[1], frame_size[0], 3), (40, 40, 40), dtype=np.uint8)
        num_landmarks = frame_landmarks.shape[0]

        # --- FIX LỖI TẠI ĐÂY ---
        # 1. Trích xuất riêng cột x và y (shape: N, 2)
        xy_coords = frame_landmarks[:, :2]
        
        # 2. Tự động kiểm tra chuẩn hóa
        is_normalized = False
        # Chỉ kiểm tra nếu mảng không bị NaN toàn bộ
        if not np.isnan(xy_coords).all():
            max_val = np.nanmax(xy_coords)
            # Nếu giá trị lớn nhất <= 2.0 (và > 0 để chắc chắn có data), nghĩa là đã bị chuẩn hóa
            if 0 < max_val <= 2.0:
                is_normalized = True
                
        scale_x = frame_size[0] if is_normalized else 1.0
        scale_y = frame_size[1] if is_normalized else 1.0
        # ---------------------------

        # 3. Vẽ các đường nối (Xương)
        for connection in self.CONNECTIONS:
            pt1_idx, pt2_idx = connection
            if pt1_idx < num_landmarks and pt2_idx < num_landmarks:
                x1, y1 = frame_landmarks[pt1_idx][:2]
                x2, y2 = frame_landmarks[pt2_idx][:2]

                # Bỏ qua nếu tọa độ bị missing/NaN hoặc mang giá trị gốc (0,0)
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2) or (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                    continue

                pt1 = (int(x1 * scale_x), int(y1 * scale_y))
                pt2 = (int(x2 * scale_x), int(y2 * scale_y))
                
                cv2.line(frame, pt1, pt2, (200, 200, 200), thickness=3, lineType=cv2.LINE_AA)

        # 4. Vẽ các điểm (Khớp)
        for i in range(num_landmarks):
            x, y = frame_landmarks[i][:2]
            if not np.isnan(x) and not np.isnan(y) and not (x == 0 and y == 0):
                center = (int(x * scale_x), int(y * scale_y))
                cv2.circle(frame, center, radius=4, color=(0, 165, 255), thickness=-1, lineType=cv2.LINE_AA)

        return frame

    def render_and_save(self, video_name: str, landmarks_sequence: np.ndarray, fps: int = 30, frame_size: Tuple[int, int] = (800, 800)) -> None:
        """
        Duyệt qua chuỗi thời gian của mảng tọa độ, render từng frame và lưu thành file .mp4.
        """
        output_path = self.output_dir / f"{video_name}.mp4"
        
        # --- LOGIC XỬ LÝ SHAPE MỚI ---
        if len(landmarks_sequence.shape) == 2:
            num_frames, total_features = landmarks_sequence.shape
            
            # Kiểm tra nếu là 75 điểm * 3 coordinates = 225
            if total_features % 3 == 0:
                num_landmarks = total_features // 3
                # Đưa mảng về dạng (số_frames, số_điểm, 3)
                landmarks_sequence = landmarks_sequence.reshape(num_frames, num_landmarks, 3)
            else:
                raise ValueError(f"Không thể parse shape {landmarks_sequence.shape}. Số lượng feature không chia hết cho 3.")
        # -----------------------------

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

        num_frames = landmarks_sequence.shape[0]
        
        for frame_idx in range(num_frames):
            frame_landmarks = landmarks_sequence[frame_idx]
            rendered_frame = self._draw_single_frame(frame_landmarks, frame_size)
            out.write(rendered_frame)

        out.release()
        logging.info(f"Đã xuất video: {output_path.name}")


# ==========================================
# KHỐI THỰC THI (MAIN)
# ==========================================
if __name__ == "__main__":
    # Cấu hình đường dẫn
    H5_FILE_PATH = "dev_data.h5"  # Thay bằng tên file thực tế của bạn
    OUTPUT_DIRECTORY = "rendered_videos"
    DATASET_KEY = "keypoints"  # Giả sử dataset bên trong mang tên này

    # 1. Khởi tạo Renderer
    renderer = SkeletonRenderer(
        h5_filepath=H5_FILE_PATH,
        output_dir=OUTPUT_DIRECTORY,
        dataset_name=DATASET_KEY
    )

    try:
        # 2. Đọc dữ liệu từ file HDF5
        logging.info("Bắt đầu đọc dữ liệu từ file HDF5...")
        all_videos_data = renderer.load_h5_data()

        # 3. Lặp qua tất cả các video (Group) và xuất file .mp4
        logging.info("Bắt đầu quá trình render video...")
        for vid_name, seq_data in all_videos_data.items():
            # seq_data dự kiến có shape (num_frames, num_landmarks, 2)
            # Mặc định sử dụng kích thước khung hình 800x800, điều chỉnh tùy theo range tọa độ của bạn
            renderer.render_and_save(
                video_name=vid_name, 
                landmarks_sequence=seq_data, 
                fps=25  # PHOENIX-2014T thường quay ở 25 fps
            )
            
        logging.info("Hoàn thành xuất toàn bộ video!")

    except FileNotFoundError as fnf_err:
        logging.error(fnf_err)
    except KeyError as k_err:
        logging.error(k_err)
    except Exception as e:
        logging.error(f"Đã xảy ra lỗi không xác định: {e}")