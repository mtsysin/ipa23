import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        # d_model must be divisible by num_heads
        assert(d_model % num_heads == 0)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        
        # Linear projections for queries, keys, values, and final outputs
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        # Reshape input tensor to be inputted into attention mechanism
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        # Permute tensor so that queries, keys, and values are in separate dimensions
        # Transpose for batch-wise computation
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, seq_len, self.d_head)
        
        return x
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        # Linearly project query, key, and value
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        # Split heads into parallel computations
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        # Scaled Dot-Product Attention
        attn_scores = torch.bmm(query, key.transpose(1, 2)) / (self.d_head ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.view(batch_size, self.num_heads, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
            attn_scores = attn_scores.view(batch_size * self.num_heads, seq_len, seq_len)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Concatenate heads and project
        context = torch.bmm(attn_weights, value)
        context = context.view(batch_size, self.num_heads, seq_len, self.d_head)
        output = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        
        return output

