'''
Implementation of SIWN backbone that will be used sometime in the future.
We'll use the original paper: https://arxiv.org/pdf/2103.14030.pdf
And borrow some insights from the original github repo
'''

import torch
import torch.nn as nn


def _overload_args(func):
    '''Helper decorator that overwrites kwargs is args from argparse are passed'''
    def wrapper(*args, **kwargs):
        if "agrs" in kwargs.keys():
            var_dict = vars(kwargs["args"])
            for key, val in var_dict.items():
                kwargs[key] = val
            del kwargs["args"]
        return func(*args, **kwargs)
    return wrapper()

def to_size(size):
    '''Converts a parameter ino a tuple of the same value if given single dim '''
    if isinstance(size, int):
        return (size, size) # For square images
    elif isinstance(size, tuple):
        return size
    else:
        raise TypeError("Unsupported size type")

class PatchEmbedding(nn.Module):
    """
    This module will cut an image into a number of patches an then apply a 
    linear layer to convert each patch into a vector of specified dimension.
    We can accept both batched an unbatched images which will help with the
    develpment process.
    """
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None) -> None:
        super().__init__()
        self.patch_size = to_size(patch_size)
        # Get the projector that takes the image and applies a convolutional layer
        # Subsequently we can reshape the result to get the desired shape
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        
    def forward(self, image):
        """
        input: image: (B), C, H, W
        output: (B), H/Ph * W/Pw, embed_dim
        """
        img_size = image.shape[-2:]
        assert ~(img_size[0]%self.patch_size[0]) and ~(img_size[1]%self.patch_size[1]), \
            "Incompatibe sizes of images and patches"
        x = self.conv(image) # (batch), embed_dim, H/Ph, W/Pw
        x = x.flatten(-1) # (batch), embed_dim, H/Ph * W/Pw
        x = x.transpose(-1, -2) # (batch), H/Ph * W/Pw, embed_dim
        if self.norm is not None:
            x = self.norm(x)

class PatchMerging(nn.Module):
    """
    This module will serve as a dimensionality reduction unit between the stages
    of the transfromer. It take 
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        # linear projector
        self.h, self.w =input_resolution
        self.linear_reduction = nn.Linear(in_features=4*dim, out_features=2*dim, bias=False)
        if norm_layer==None:
            self.norm = lambda x: x

    def forward(self, x: torch.Tensor):
        """
        input: x: (B), H*W, embed_dim
        From paper:
        patch merging layer concatenates the features of each group of 2×2 neighboring patches, 
        and applies a linear layer on the 4C - dimensional concatenated features. 
        This reduces the number of tokens by a multiple of 2×2 = 4 (2× downsampling of resolution), 
        and the output dimension is set to 2C.
        output: (B), H*W/4, 2C
        """
        l, c = x.shape[-2:]
        assert l == self.h * self.w, "input feature has wrong size"
        assert self.h % 2 == 0 and self.w % 2 == 0, f"x size ({self.h}*{self.w}) are not even."
        if len(x.shape) > 2:
            batch_dims = x.shape[:-2]
        else:
            batch_dims = []
        x = x.view(*batch_dims, self.h, self.w, c) # Convert back to 2d representation
        # The next line splits the image into 2x2 patches and then extracts each "corner"
        # to get the size of (B) H/2 W/2 C. Then we concatenate all that
        x = torch.cat([x[..., i::2, j::2, :] for i in range(2) for j in range(2)], -1)  # (b) H/2 W/2 4*C
        x = x.view(*batch_dims, -1, 4 * c)  # (B) H/2*W/2 4*C
        x = self.norm(x)
        return self.linear_reduction(x)



# In this block we'll try to implement the logic for shifted window attention mechanism.

# The following block is directly copied from the original repo with more comments attached.
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    most of the inputs are the usual attention mechanism parameters which we will not discuss to much further.
    inputs:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., 
                 initialization = {
                     "name": nn.init.trunc_normal_,
                     "kwargs": {"std": .02}
                    }
                ):

        super().__init__()
        # Unpack stuff
        self.dim = dim 
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # There are 2 * Wh - 1 distances possible in the window of size Wh.
        # This table is alearnable parameter from which we can draw values for the final matrix b
        # where the index will solely rely on the relative position between the two patches in the
        # window.
        # Number of heads at the end is essentially needed to make sure that for each head we have
        # a different set of learnable parameters.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # This is the most interesting part of this block. Essentially, it tries to
        # build a pair-wise relative position index for each token inside the window
        # This means that we manually create a tensor which has all possible patches on one side (Wh)
        # and all possible matching patces on another side (Wh). For each such pair we create two numbers:
        # X and Y dimension distance.
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # shift to start from 0, beacuse we want to use these values as indices.
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # We also want to differentiate between the horizontal and ertical coordiantes.
        # Thus essentially we crate a base-(2 * self.window_size[1] - 1) number with the two coordinates.
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Standard qkv decomposition stuff, exactly as in Attention is all you need.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize the relative position bias table with chosen intialization function
        # (truncated normal by default)
        initialization['name'](self.relative_position_bias_table, **initialization['kwargs'])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def make_window_partition(image: torch.Tensor, window_size):
    '''
    Taken an image and retruns a partitioned version,
    Accepts both batched and unbatched inputs
    input: image: (B), H, W, C
        patch_size (int): patch size

    '''
    H, W, C = image.shape[-3:]
    if len(image.shape) > 3:
        batch_dims = image.shape[:3]
    else:
        batch_dims = []
    x = x.view(*batch_dims, H // window_size, window_size, W // window_size, window_size, C)
    l_bdims = len(batch_dims)
    return x.permute(*range(l_bdims), *[l_bdims + i for i in [0, 2, 1, 3, 4]])\
        .contiguous().view(-1, window_size, window_size, C), batch_dims



def window_reverse(windows, window_size, H, W, batch_dims):
    """
    Essentially reverses the window partition operation
    input:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        batch_dims: stored batch dims to reconstruct the tensor

    Returns:
        x: (B), H, W, C
    """
    x = windows.view(*batch_dims, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(*batch_dims, H, W, -1)


class SWINBlock(nn.Module):

    @_overload_args
    def __init__(self, model_capcaity) -> None:
        super().__init__()


    def forward():
        '''
        Forward description
        '''
        # convert the input image ot
