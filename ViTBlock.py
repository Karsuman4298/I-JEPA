import torch

class ViTBlock(torch.nn.Module):

    def __init__(self, embedding_dim: int, mlp_embedding_dim_expension: int, attention_heads: int):
        super().__init__()

        self.input_layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.hidden_layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        self.attention = torch.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True)

        self.mlp1 = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim * mlp_embedding_dim_expension, bias=True) 
        self.mlp2 = torch.nn.Linear(in_features=embedding_dim * mlp_embedding_dim_expension, out_features=embedding_dim, bias=True) 
        self.gelu = torch.nn.GELU()

    def forward(self, x, key_padding_mask=None):

        # Normalize input
        input_normalized = self.input_layer_norm(x)

        # Prepare key_padding_mask: shape (B, N), True = ignore
        if key_padding_mask is not None:        
            key_padding_mask = ~key_padding_mask  # invert: True means ignore
        
            # If all positions are masked for a batch element, unmask one
            if key_padding_mask.all(dim=1).any():
                key_padding_mask[:, 0] = False

        # Calculate attention values with mask
        attention_output, _ = self.attention(input_normalized, input_normalized, input_normalized, key_padding_mask=key_padding_mask)

        # Zero out outputs for masked queries 
        if key_padding_mask is not None:
            attention_output = attention_output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # Apply residual connection here. Add the input to the attention output again, and run it through the second layer norm
        hidden_state = attention_output + x
        
        mlp_output = self.gelu(self.mlp2(self.mlp1(self.hidden_layer_norm(hidden_state))))

        # Apply last residual connection
        return mlp_output + hidden_state