import torch
from einops import rearrange

from ViT22B_Block import ViT_22B_Block
from ViTBlock import ViTBlock

class TargetEncoder(torch.nn.Module):

    # These are small values, so we can train it on standard hardware and datasets
    def __init__(self, block_count = 8, embedding_dim = 128, attention_heads = 4, mlp_expension = 4, patch_size=16, patch_count_per_side = 14):
        super().__init__()

        self.block_count = block_count
        self.patch_size = patch_size
        self.patch_count_per_side = patch_count_per_side
        self.embedding_dim = embedding_dim

        # Create projection layer to project image patches to tokens
        self.patch_projection = torch.nn.Linear(in_features=patch_size**2 * 3, out_features=embedding_dim)

        # Create ViT-22 transformer blocks
        self.transformer_blocks = torch.nn.ModuleList([
            ViTBlock(embedding_dim=embedding_dim, mlp_embedding_dim_expension=mlp_expension, attention_heads=attention_heads)
            for _ in range(block_count)
        ])
        
        # Create learnable positional encoding
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, patch_count_per_side**2, embedding_dim))

        # Output layer norm to normalize output tokens
        self.output_layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, img):

        B, C, H, W = img.shape
        assert H == W == self.patch_count_per_side * self.patch_size
        
        # Calcualte total number of patches
        num_patches = self.patch_count_per_side ** 2

        # Divide image into non overlapping patches
        patches = rearrange(img, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=self.patch_size, pw=self.patch_size)

        # Linear projection of patches to get the input tokens
        tokens = self.patch_projection(patches)

        # Add positional embedding to the tokens
        tokens = tokens + self.pos_embedding[:, :num_patches].expand(B, -1, -1)

        # Run tokens throught he encoder
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Reshape tokens back to (batch_size, height, width, embedding_dim)
        # sequence_length = num_patches = self.patch_count_per_side ** 2
        tokens = rearrange(tokens, 'b (h w) d -> b h w d', h=self.patch_count_per_side)

        # Return tokens
        return self.output_layer_norm(tokens)