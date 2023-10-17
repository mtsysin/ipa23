import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

class DETREncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(DETREncoder, self).__init__()

        self.c1 = DETREncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.c2 = DETREncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.c3 = DETREncoderLayer(d_model, nhead, dim_feedforward, dropout)

    def forward(self, x, mask=None):
        # Apply the layers sequentially
        out = self.c1(x, src_key_padding_mask=mask)
        out = self.c2(out, src_key_padding_mask=mask)
        out = self.c3(out, src_key_padding_mask=mask)
        return out


class DETREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(DETREncoderLayer, self).__init()
        
        # Multi-Head Self-Attention
        self.multihead_atten = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # Layer normalization for self-attention output
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization for feedforward output
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Multi-Head Self-Attention
        attn = self.multihead_atten(x, x, x, mask=mask)
        
        # Residual connection and layer normalization
        res = x + attn
        norm = self.norm1(res)
        
        # Feedforward layer
        out = self.feedforward(norm)
        
        # Residual connection and layer normalization
        res2 = x + out
        out2 = self.norm2(res2)
        
        return out2
