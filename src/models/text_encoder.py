"""
Transformer Encoder để mã hóa chuỗi Glosses.

Dựa trên kiến trúc Transformer (Vaswani et al., 2017) với Positional Encoding.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding chuẩn của Transformer.
    
    Tham khảo: "Attention is All You Need" (Vaswani et al., 2017)
    https://arxiv.org/abs/1706.03762
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model (int): Dimension của embedding
            max_len (int): Độ dài tối đa của sequence
            dropout (float): Tỷ lệ dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Tạo ma trận PE shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape [batch, seq_len, d_model]
            
        Returns:
            Tensor với positional encoding đã cộng vào
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTextEncoder(nn.Module):
    """
    Transformer Encoder để mã hóa chuỗi Glosses thành context vector.
    
    Architecture:
        Embedding -> Positional Encoding -> Transformer Layers -> Pooling
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        """
        Args:
            vocab_size (int): Kích thước vocabulary
            d_model (int): Dimension của hidden state
            nhead (int): Số attention heads
            num_layers (int): Số lớp Transformer
            dim_feedforward (int): Dimension của FFN
            dropout (float): Dropout rate
            max_len (int): Độ dài tối đa input sequence
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input shape: [batch, seq, feature]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(
        self, 
        text_ids: torch.Tensor, 
        text_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encode chuỗi glosses thành context vector.
        
        Args:
            text_ids: Tensor shape [batch, seq_len] - tokenized glosses
            text_mask: Tensor shape [batch, seq_len] - mask (1=valid, 0=pad)
            
        Returns:
            context: Tensor shape [batch, d_model] - vector ngữ cảnh
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, d_model]
        x = self.embedding(text_ids) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Tạo attention mask cho Transformer (True = ignore)
        if text_mask is not None:
            # Đảo mask: text_mask có 1=valid, nhưng Transformer cần True=padding
            attn_mask = ~text_mask  # Shape: [batch, seq_len]
        else:
            attn_mask = None
        
        # Transformer encoding: [batch, seq_len, d_model]
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # ========== FIX: Masked mean pooling với dimension matching ==========
        if text_mask is not None:
            # Lấy chiều dài thực tế của encoded (có thể bị truncate bởi Transformer)
            actual_seq_len = encoded.size(1)
            
            # Cắt mask theo chiều dài thực tế
            mask_truncated = text_mask[:, :actual_seq_len]  # [batch, actual_seq_len]
            mask_expanded = mask_truncated.unsqueeze(-1).float()  # [batch, actual_seq_len, 1]
            
            # Masked sum
            sum_encoded = (encoded * mask_expanded).sum(dim=1)  # [batch, d_model]
            count = mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
            context = sum_encoded / count
        else:
            context = encoded.mean(dim=1)  # [batch, d_model]
        
        return context