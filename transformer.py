import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder transformer architecture
    """
    def __init__(self, encoder, decoder, x_embed, tgt_embed, linear):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.x_embed = x_embed
        self.tgt_embed = tgt_embed
        self.linear = linear
        
    def forward(self, x, out, x_mask, out_mask):
        """
        Take in and process masked input and output sequences.
        Send input and mask through encoder, send result, output, and output mask through decoder
        """
        x = self.encode(x, x_mask)
        out = self.decode(x, x_mask, out, out_mask)
        return out
    
    def encode(self, x, x_mask):
        """
        Send input through encoder stack
        """
        out = self.encoder(self.x_embed(x), x_mask)
        return out
    
    def decode(self, x_saved, x_mask, out, out_mask):
        """
        Send input,  through decoder stack
        """
        out = self.decoder(self.tgt_embed(out), x_saved, x_mask, out_mask)
        return out
    

class Linear(nn.Module):
    """
    Linear + softmax generation step
    """
    def __init__(self, d_model, vocab):
        super(Linear, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    


def clone_layers(module, N):
    """
    Clone N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    """
    Layer normalization module
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.ones = nn.Parameter(torch.ones(features))
        self.zeros = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        Get mean and std for normalization
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = self.ones * (x - mean) / (std + self.eps) + self.zeros
        return out
    

class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to sublayer with same size
        Carry out dropout on sublayer output
        """
        x_norm = self.norm(x)
        sub = sublayer(x_norm)
        residual = x + self.dropout(sub)
        return residual



class Encoder(nn.Module):
    """
    Encoder with N layers
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone_layers(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, x_mask):
        """
        Pass the input and mask through each layer
        """
        for layer in self.layers:
            x = layer(x, x_mask)
        out = self.norm(x)
        return out