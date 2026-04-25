"""
PyTorch Dataset cho PHOENIX-2014T.

Load dữ liệu keypoints từ file HDF5 và tokenize glosses thành tensor.
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
from .vocabulary import GlossVocabulary


class PhoenixDataset(Dataset):
    """
    Dataset class cho PHOENIX-2014T Sign Language Production.
    
    Attributes:
        h5_path (str): Đường dẫn đến file .h5 chứa keypoints
        vocab (GlossVocabulary): Vocabulary object để encode glosses
        max_len (int): Độ dài tối đa của chuỗi text (để padding)
    """
    
    def __init__(
        self, 
        h5_path: str, 
        vocab: GlossVocabulary,
        max_text_len: int = 100,
        max_pose_len: int = 300
    ):
        """
        Khởi tạo dataset.
        
        Args:
            h5_path (str): Đường dẫn file HDF5 (train/dev/test_data.h5)
            vocab (GlossVocabulary): Vocab đã được build sẵn
            max_text_len (int): Độ dài tối đa chuỗi gloss (padding)
            max_pose_len (int): Độ dài tối đa chuỗi pose (padding/truncate)
        """
        self.h5_path = h5_path
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.max_pose_len = max_pose_len
        
        # Đọc metadata từ HDF5
        with h5py.File(h5_path, 'r') as f:
            self.video_ids = list(f.keys())
        
        print(f"📂 Loaded {len(self.video_ids)} samples from {h5_path}")
    
    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset."""
        return len(self.video_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Lấy một sample từ dataset.
        
        Args:
            idx (int): Chỉ số sample
            
        Returns:
            Dict chứa:
                - 'text_ids': Tensor shape [max_text_len] (tokenized glosses)
                - 'text_mask': Tensor shape [max_text_len] (1=valid, 0=padding)
                - 'pose': Tensor shape [max_pose_len, 225] (keypoints)
                - 'pose_mask': Tensor shape [max_pose_len] (1=valid, 0=padding)
                - 'text_len': Độ dài thực của text
                - 'pose_len': Độ dài thực của pose
        """
        video_id = self.video_ids[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            group = f[video_id]
            keypoints = group['keypoints'][:]  # Shape: [T, 225]
            gloss_text = group.attrs['label']  # String glosses
        
        # --- Xử lý Text ---
        text_ids = self.vocab.encode(gloss_text, add_special=True)
        text_len = len(text_ids)
        
        # Padding hoặc truncate
        if text_len > self.max_text_len:
            text_ids = text_ids[:self.max_text_len]
            text_len = self.max_text_len
        
        text_mask = [1] * text_len + [0] * (self.max_text_len - text_len)
        text_ids = text_ids + [self.vocab.pad_id] * (self.max_text_len - text_len)
        
        # --- Xử lý Pose ---
        pose_len = keypoints.shape[0]
        
        if pose_len > self.max_pose_len:
            keypoints = keypoints[:self.max_pose_len]
            pose_len = self.max_pose_len
        
        pose_mask = [1] * pose_len + [0] * (self.max_pose_len - pose_len)
        
        # Padding pose với zero vector
        if pose_len < self.max_pose_len:
            padding = np.zeros((self.max_pose_len - pose_len, 225), dtype=np.float32)
            keypoints = np.vstack([keypoints, padding])
        
        return {
            'text_ids': torch.LongTensor(text_ids),
            'text_mask': torch.BoolTensor(text_mask),
            'pose': torch.FloatTensor(keypoints),
            'pose_mask': torch.BoolTensor(pose_mask),
            'text_len': text_len,
            'pose_len': pose_len
        }