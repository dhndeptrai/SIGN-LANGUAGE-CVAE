"""
Vocabulary Builder for PHOENIX-2014T Glosses.

Xây dựng từ điển ánh xạ từ chuỗi Glosses (tiếng Đức) sang ID số.
Hỗ trợ các token đặc biệt: <pad>, <sos>, <eos>, <unk>.
"""

import pickle
import json
from collections import Counter
from typing import List, Dict, Optional


class GlossVocabulary:
    """
    Quản lý vocabulary cho chuỗi Glosses.
    
    Attributes:
        word2id (Dict[str, int]): Ánh xạ từ gloss -> ID
        id2word (Dict[int, str]): Ánh xạ ngược ID -> gloss
        special_tokens (Dict[str, str]): Các token đặc biệt
    """
    
    def __init__(self, min_freq: int = 2):
        """
        Khởi tạo vocabulary với tần suất tối thiểu.
        
        Args:
            min_freq (int): Từ xuất hiện ít hơn min_freq sẽ được xem là <unk>
        """
        self.min_freq = min_freq
        self.special_tokens = {
            'pad': '<pad>',
            'sos': '<sos>',  # Start of sequence
            'eos': '<eos>',  # End of sequence
            'unk': '<unk>'   # Unknown token
        }
        
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        
        # Thêm special tokens vào đầu vocab
        for token in self.special_tokens.values():
            self._add_word(token)
    
    def _add_word(self, word: str) -> int:
        """
        Thêm một từ vào vocabulary.
        
        Args:
            word (str): Gloss cần thêm
            
        Returns:
            int: ID của từ vừa thêm
        """
        if word not in self.word2id:
            idx = len(self.word2id)
            self.word2id[word] = idx
            self.id2word[idx] = word
            return idx
        return self.word2id[word]
    
    def build_from_sentences(self, sentences: List[str]) -> None:
        """
        Xây dựng vocabulary từ danh sách câu.
        
        Args:
            sentences (List[str]): Danh sách chuỗi glosses (cách nhau bởi khoảng trắng)
            
        Example:
            >>> vocab = GlossVocabulary(min_freq=2)
            >>> sentences = ["HEUTE MORGEN", "MORGEN REGEN"]
            >>> vocab.build_from_sentences(sentences)
        """
        # Đếm tần suất từ
        word_counter = Counter()
        for sentence in sentences:
            words = sentence.strip().split()
            word_counter.update(words)
        
        # Chỉ thêm từ có tần suất >= min_freq
        for word, freq in word_counter.items():
            if freq >= self.min_freq:
                self._add_word(word)
        
        print(f"✅ Vocabulary built: {len(self.word2id)} tokens (min_freq={self.min_freq})")
    
    def encode(self, sentence: str, add_special: bool = True) -> List[int]:
        """
        Chuyển câu thành danh sách ID.
        
        Args:
            sentence (str): Chuỗi glosses
            add_special (bool): Có thêm <sos> và <eos> không
            
        Returns:
            List[int]: Danh sách ID tương ứng
        """
        words = sentence.strip().split()
        ids = [self.word2id.get(w, self.word2id['<unk>']) for w in words]
        
        if add_special:
            ids = [self.word2id['<sos>']] + ids + [self.word2id['<eos>']]
        
        return ids
    
    def get_id(self, word: str) -> int:
        """
        Return the ID for a gloss token, or the <unk> ID if not found.

        Args:
            word (str): Gloss token.

        Returns:
            int: Token ID.
        """
        return self.word2id.get(word, self.word2id[self.special_tokens['unk']])
    
    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        """
        Chuyển danh sách ID thành câu.
        
        Args:
            ids (List[int]): Danh sách ID
            remove_special (bool): Loại bỏ các token đặc biệt
            
        Returns:
            str: Chuỗi glosses gốc
        """
        words = [self.id2word.get(i, '<unk>') for i in ids]
        
        if remove_special:
            words = [w for w in words if w not in self.special_tokens.values()]
        
        return ' '.join(words)
    
    def save(self, save_dir: str) -> None:
        """Lưu vocabulary ra file pickle và JSON."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/gloss_vocab.pkl", 'wb') as f:
            pickle.dump(self, f)
        
        with open(f"{save_dir}/gloss_to_id.json", 'w', encoding='utf-8') as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved vocab to {save_dir}/")
    
    @staticmethod
    def load(vocab_path: str) -> 'GlossVocabulary':
        """Load vocabulary từ file pickle."""
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)
    
    def __len__(self) -> int:
        """Trả về số lượng token trong vocab."""
        return len(self.word2id)
    
    @property
    def pad_id(self) -> int:
        """ID của token <pad>."""
        return self.word2id['<pad>']