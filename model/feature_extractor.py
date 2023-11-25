from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from ddp import is_main_process
from positional_embedding import build_position_encoding, NestedTensor


'''
Batch normalization where the running mean and variance are fixed during training
To use pre-trained weights in the backbone, we'll just directly copy-paste the definitions fom the
original implementation. This is because when loading from state dict, the pytorch engine registers 
module names as literally the names of the variables in python.
'''
class FrozenBatchNorm2d(torch.nn.Module):
    '''
    Here is the frozen batchnorm. The issue with transformer-like architectures is that it's very hard to use
    batchnorm donce the shapes of inpuths are often different. Thus to free ourselves form the pain, we use
    batchnorm with frozen statistics. This iessentailly keeps the module in .eval() mode for the whole training
    process.
    In the develpment process, this thing just replaces the 
    '''
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # Parameters for batch normalization
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Exclude 'num_batches_tracked' from the state_dict to align with frozen batch norm
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # Calculate normalization parameters
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

'''
Base class for the backbone of the model
'''
class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()

        # Freeze certain layers of the backbone if `train_backbone` is False
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        # Define layers to return intermediate feature maps if required
        # This not really needed for us, but we keep it to prevent bugs when loading the model.
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        # Extract intermediate features from backbone
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        
        # Resize the mask to match the feature map sizes and create NestedTensor
        # xs here is a list of outputs form the model at either the last layer,
        # or intermediate layers as shown in the __init__ method
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # We reinterploate the mask to matchthe new feature size. The is essentailly bcause
            # The CNN-like architectures o the backbone are spatially invariantand we can
            # Easiily figure out that the resulting features corresponding to the portion of the image that was 
            # masked out is not relevant and should be masked out as well.
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

'''
Backbone class defining the feature extraction architecture
'''
class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        # Load the selected backbone model
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

'''
Module to combine backbone features with positional encoding
'''
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        #self[0] and self[1] refer to the baclbone and the encoder respectively.
        # This is just inherited from nn.Sequential.
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # Generate positional embeddings for each feature map
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

'''
Function to build the backbone model based on input arguments
'''
def build_backbone(args):
    # Build positional encoding
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    # Construct the backbone with intermediate feature layers if required
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # Combine backbone features with positional embeddings
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model