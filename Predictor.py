import torch
from einops import rearrange

from ViTBlock import ViTBlock

class Predictor(torch.nn.Module):

    # These are small values, so we can train it on standard hardware and datasets
    def __init__(self, block_count = 8, embedding_dim = 128,  attention_heads = 4, mlp_expension = 4, patch_size=16, patch_count_per_side = 14):
        super().__init__()

        self.block_count = block_count
        self.patch_size = patch_size
        self.patch_count_per_side = patch_count_per_side
        self.embedding_dim = embedding_dim

        # Create projection layer to project image patches to tokens
        self.patch_projection = torch.nn.Linear(in_features=patch_size**2 * 3, out_features=embedding_dim)

        # Create ViT-22 transformer blocks
        self.transformer_blocks = torch.nn.ModuleList([
            ViTBlock(embedding_dim=embedding_dim, mlp_embedding_dim_expension = mlp_expension, attention_heads=attention_heads)
            for _ in range(block_count)
        ])
        
        # Create learnable positional encoding
        self.context_pos_encoding = torch.nn.Parameter(torch.zeros(1, patch_count_per_side**2, embedding_dim))
        
        # Create the learnable mask token
        self.predictor_mask_token = torch.nn.Parameter(torch.zeros(1, patch_count_per_side**2, embedding_dim))

        # Output layer norm to normalize output tokens
        self.output_layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, context_representation, context_token_mask, target_block_mask):
        B, S, D = context_representation.shape
        device = context_representation.device

        # Positional encoding
        context_positional_encoding = self.context_pos_encoding.expand(B, -1, -1)

        context_representation = context_representation + context_positional_encoding
        mask_tokens = self.predictor_mask_token.expand(B, -1, -1) + context_positional_encoding

        context_representation = torch.cat((context_representation, mask_tokens), dim=1)  # [B, S*2, D]

        # Build complete token mask
        complete_token_mask = torch.zeros((B, S * 2), dtype=torch.bool, device=device)
        for b in range(B):
            mask = torch.zeros(S, dtype=torch.bool, device=device)
            for y in range(target_block_mask[b][0], target_block_mask[b][2]):
                for x in range(target_block_mask[b][1], target_block_mask[b][3]):
                    mask[y * self.patch_count_per_side + x] = True

            complete_token_mask[b, :S] = context_token_mask[b]
            complete_token_mask[b, S:] = mask

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            context_representation = block(context_representation, complete_token_mask)

        return self.output_layer_norm(context_representation)
