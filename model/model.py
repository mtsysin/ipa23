import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from typing import Union, Optional
from torch import Tensor
import copy
from misc import nested_tensor_from_tensor_list, NestedTensor
from model.feature_extractor import build_backbone

"""Most of the stuff is directly or indiectly taken fomr the original implementation"""

def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

class FFN(nn.Module):
    """ Very simple FFN mainly used to calculate box embeddings"""

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

def build_detr(args):

    backbone = build_backbone(args)
    transformer = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

    model = DETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        # TODO: figure out auxilary loss later
        aux_loss=False #args.aux_loss,
    )

    return model


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # Inidialize the embeddings that take trhe outputs of the transofrmer and return the class vector
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # Initializes a simple FFN that takes the output of the transformer and creates the values for the bounding boxes
        self.bbox_embed = FFN(hidden_dim, hidden_dim, 4, 3)
        # This is for the initilization of  the query vector. We create an embedding and act on it with th 
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: Union[NestedTensor, list, torch.Tensor]):
        """Input: NestedTensor:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
                                                                        that were artificially added to make images the same size

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size, num_queries, (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        # Convert to nested tensor if necessary
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        # Get the last output of the (intermediate) feature fom the feature extractor
        src, mask = features[-1].decompose()
        assert mask is not None
        # Get the output of the decoder, shape num_intermediate, batch, num_queries, d_model
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # Get class embeddings: 
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # Not really needed
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_block = DETREncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, False, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(num_encoder_layers, encoder_block, encoder_norm)

        decoder_block = DETRDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, False, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(num_decoder_layers, decoder_block, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Funciton to reinitialize parameters with Xabier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        '''
        Main run of transformer
        Takes feature of shape:
        src: (batch, channel, h, w) (h, w, -- padded height and wifth)
        mask: (batch, h, w), -- associated mask for the image
        query_embed: ()
        '''
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
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(num_layers)])  # Stack together multiple 
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
        super(DETREncoderLayer, self).__init__()
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

        self.activation_func = activation_dict[activation_func]

    @staticmethod
    def add_embedding(x, pos):
        """ Adds positional embedding to the input tensor x"""
        return x if pos is None else x + pos

    def forward(self, src,
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
        super(DETRDecoderLayer, self).__init__()
        self.normalize_before = normalize_before
        self.activation_func = activation_dict[activation_func]
        
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

    
    @staticmethod
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
                                    value=memory, attn_mask=memory_mask,
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