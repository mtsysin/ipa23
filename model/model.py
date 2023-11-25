import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from typing import Optional
from torch import Tensor
import copy

def mish(self, x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

class FFN(nn.Module):
    """ Very simple FFN """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation = F.relu):
        super().__init__()
        self.activation = activation
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# class DETREncoder(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super(DETREncoder, self).__init__()

#         self.c1 = DETREncoderLayer(d_model, nhead, dim_feedforward, dropout)
#         self.c2 = DETREncoderLayer(d_model, nhead, dim_feedforward, dropout)
#         self.c3 = DETREncoderLayer(d_model, nhead, dim_feedforward, dropout)

#     def forward(self, x, mask=None):
#         # Apply the layers sequentially
#         out = self.c1(x, src_key_padding_mask=mask)
#         out = self.c2(out, src_key_padding_mask=mask)
#         out = self.c3(out, src_key_padding_mask=mask)
#         return out


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = DETREncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = DETRDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)



class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, encoder_block, norm=None):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(num_layers)])  # Stack together multiple 
                                                                                                # encoder blocks
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        # Apply all layers sequentially
        for layer in self.layers:
            output = layer(output, 
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, 
                           pos=pos)

        return self.norm(output) if self.norm is not None else output


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, decoder_block, norm=None, return_intermediate=False):
        super().__init__()
        nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(num_layers)])  # Stack together multiple 
                                                                                                # encoder blocks
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            # Update the last intermediate if we used norm at the end
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class DETREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 use_custom_attention = False, 
                 activation_func = 'mish',
                 normalize_before = False
        ):
        """
        The meat of the encoder nework. Pretty much follows the picture form the paper
        
        """
        super(DETREncoderLayer, self).__init()
        self.normalize_before = normalize_before
        self.activation_func = activation_func
        
        # Multi-Head Self-Attention -- initialize the attention mechanism -- can choose between the build-in version as well as 
        # the custim one
        if use_custom_attention:
            self.multihead_atten = MultiHeadAttention(d_model, nhead, dropout=dropout)
        else:
            self.multihead_atten = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Layer normalization for self-attention output
        self.norm1 = nn.LayerNorm(d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation_finc = activation_dict[activation_func]

    
    def add_embedding(x, pos):
        """ Adds positional embedding to the input tensor x"""
        return x if pos is None else x + pos

    def forward(self, x,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """Performs a forward pass of an encoder layer with two options as described in the original impoementation"""

        # Normalizes inputs as shown in the original attention os all you need.
        if self.normalize_before:
            src2 = self.norm1(src)
            q = k = self.add_embedding(src2, pos)
            src2 = self.multihead_atten(q, k, value=src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src) # Normalizing before entering the second stage of the encoder module.
            # No idea why they don't add an activatoin funciton after second linear layer
            # Anyway, it's just a simple FFN with two layers
            src2 = self.linear2(self.dropout(self.activation_func(self.linear1(src2))))
            src = src + self.dropout2(src2)

        # Unorthodox verison. Adds embeddings before normalizing the input layer.
        else:
            q = k = self.add_embedding(src, pos)
            src2 = self.multihead_atten(q, k, value=src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation_func(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        return src


class DETRDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 use_custom_attention = False, 
                 activation_func = None,
                 normalize_before = False
        ):
        super(DETREncoderLayer, self).__init()
        self.normalize_before = normalize_before
        self.activation_finc = activation_dict[activation_func]
        
        # Multi-Head Self-Attention
        if use_custom_attention:
            self.multihead_atten = MultiHeadAttention(d_model, nhead, dropout=dropout)
            self.self_attention = MultiHeadAttention(d_model, nhead, dropout=dropout)
        else:
            self.multihead_atten = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Layers for decoder block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    
    def add_embedding(x, pos):
        """ Adds positional embedding to the input tensor x"""
        return x if pos is None else x + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """Performs a forward pass of a decoder layer with two options as described in the original impoementation"""
        if self.normalize_before:
            tgt2 = self.norm1(tgt)
            q = k = self.add_embedding(tgt2, query_pos)
            tgt2 = self.self_attention(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_atten(query=self.add_embedding(tgt2, query_pos),
                                    key=self.add_embedding(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.add_embedding(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else: 
            q = k = self.add_embedding(tgt, query_pos)
            tgt2 = self.self_attention(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.multihead_atten(query=self.add_embedding(tgt, query_pos),
                                    key=self.add_embedding(memory, pos),
                                    value=memory, attn_add_embeddingmask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation_func(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
        return tgt

activation_dict = {
    'relu': F.relu,
    'gelu': F.gelu,
    'glu': F.glu,
    'mish': F.mish,
}