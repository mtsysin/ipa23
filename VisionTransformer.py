import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
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
        
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        # Reshape input tensor to be inputted into attention mechanism
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        # Permute tensor so that queries, keys, and values are in separate dimensions
        x = x.permute(0, 2, 1, 3)
        return x
    
    def forward(self, query, key, value, mask=None):
        # Linearly project query, key, and value
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        # Split heads into parallel computations
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        # Concatenate heads and project
        output = output.permute(0, 2, 1, 3).contiguous().view(-1, query.size(2), self.d_model)
        output = self.W_o(output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Two linear layers with ReLU activation in between
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, d_model, num_heads, num_layers, hidden_dim):
        super(VisionTransformer, self).__init__()

        # Patch Embedding Layer
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttention(d_model, num_heads),
                FeedForward(d_model, hidden_dim)
            ])
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Patch embedding and reshaping
        x = self.patch_embed(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)

        # Add positional embeddings and apply layer normalization
        x += self.positional_embedding
        x = self.layer_norm(x)

        # Transformer layers
        for attn, ff in self.layers:
            x = x + attn(x, x, x)
            x = x + ff(x)

        # Global average pooling and classification head
        x = x.permute(0, 2, 1)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Hyperparameters:
image_size = (640, 360)  # Input image size (width, height)
patch_size = 20   # Patch size
num_classes = 10  # Number of output classes
d_model = 768   # Embedding dimension
num_heads = 12    # Number of attention heads
num_layers = 12   # Number of transformer layers
hidden_dim = 3072 # Hidden layer dimension

# Create a ViT model
vit_model = VisionTransformer(image_size, patch_size, num_classes, d_model, num_heads, num_layers, hidden_dim)

# Print the model architecture
print(vit_model)

# Output
'''
VisionTransformer(
  (patch_embed): Conv2d(3, 768, kernel_size=(20, 20), stride=(20, 20))
  (layers): ModuleList(
    (0-11): 12 x ModuleList(
      (0): MultiHeadAttention(
        (W_q): Linear(in_features=768, out_features=768, bias=True)
        (W_k): Linear(in_features=768, out_features=768, bias=True)
        (W_v): Linear(in_features=768, out_features=768, bias=True)
        (W_o): Linear(in_features=768, out_features=768, bias=True)
      )
      (1): FeedForward(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
      )
    )
  )
  (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (fc): Linear(in_features=768, out_features=10, bias=True)
)
'''