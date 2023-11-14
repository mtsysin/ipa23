import math
import torch
from torch import nn, Tensor
from typing import Optional, List



"""
Nested Tensor class
Source: facebookresearch/detr
"""
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


"""
From the DETR paper:
'Since the transformer architecture is permutation-invariant, we supplement it with fixed positional
encodings that are added to the input of each attention layer. We defer to the supplementary material
the detailed definition of the architecture, which follows the one described in Attention Is All You Need.
There are two kinds of positional encodings in our model: spatial positional encodings and output positional
encodings (object queries).'

From Annotated DETR:
'The spatial encodings refer to the spatial positions H,W in the lower resolution feature images C x H x W
The output encodings refer to the positions of the objects in the image and are always learned'

Thus each attention layer must have a positional encoding matrix added to it (element-wise addition)
These learned weight matrices will supplement the feature tokens with spatial context
"""
class PositionEmbeddingSine(nn.Module):
    """
    From "Attention is all you need", the fixed positional encodings are predefined as:
    - PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    Where pos is the position and i is the dimensionality

    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        # If a scale is provided and 'normalize' is False, raise an error
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        
        # If scale is not provided, set it to 2pi
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        # Create a mask indicating the positions that are not masked
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None

        # Create a mask indicating the positions that are not masked using bitwise inverse
        not_mask = ~mask

        # Calculate the cumulative sum along the height and width dimensions
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6

            # Normalize the cumulative sums and scale them
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Generate a sequence of numbers for positional encoding       
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Calculate positional encoding for x and y coordinates
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Apply sine and cosine functions to positional encodings
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # Concatenate the positional encodings for x and y
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

"""
Class for generating learned positional embeddings for input tensors

Create learned positional embeddings that can be added to input tensors. These positional
embeddings provide information about the spatial location of elements in the input tensor

Create embedding layers for both row and column positions as well as a method to reset the
embedding parameters with a uniform distribution

The forward pass takes a tensor_list as input, which contains tensors and masks. It extracts
the input tensors, computes row and column embeddings, and concatenates them to generate
the final positional embeddings that match the input tensor's shape
"""
class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=256):
        super().__init__()
        # Create an embedding layer for rows and columns with num_pos_feats dimensions
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

        # Initialize the embedding parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the embedding weights from a uniform distribution
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        # Extract the tensors from the input tensor_list
        x = tensor_list.tensors

        # Get the height h and width w dimensions of the tensor
        h, w = x.shape[-2:]

        # Create tensors to represent row and column indices
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        # Embed row and column indices using the embedding layers
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        # Concatenate the row and column embeddings and repeat them to match the input tensor shape
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

"""
Function to create position encodings for a given model configuration

This function is responsible for building and returning position encodings based on the specified
configuration arguments. Position encodings provide spatial information to the model

Function calculates the number of steps N_steps for the position encoding based on the
specified hidden dimension. It then selects the appropriate method for generating position encodings
based on the args.position_embedding argument ('v2' or 'v3') and returns the corresponding position
embedding module
"""
def build_position_encoding(args):
    N_steps = args.hidden_dim // 2

    # Choose the position embedding method based on the configuration argument
    if args.position_embedding in ('v2', 'sine'):
        # Create a Sine-based position embedding with N_steps features and normalization
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        # Create a learned position embedding with N_steps features
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        # Raise an error for unsupported position embedding methods
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding






