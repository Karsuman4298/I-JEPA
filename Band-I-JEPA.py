import math
import torch
import torch.nn as nn
from einops import rearrange
from ViTBlock import ViTBlock
# ─────────────────────────────────────────────
# MODULE A: Band-Axis Spectral Gating (NEW)
#   1-D FFT along the wavelength/band axis.
#   Sits BEFORE patch projection so the raw
#   per-band structure is still intact.
# ─────────────────────────────────────────────
class BandSpectralGatingNetwork(nn.Module):
    """
    Input  : (B, N, P, P, C)  – N patches, each P×P pixels, C hyperspectral bands
    Output : (B, N, P, P, C)  – same shape, band -axis filtered + residual
    """
    def __init__(self, num_bands: int):
        super().__init__()
        self.num_bands = num_bands
        num_freq_bins = num_bands // 2 + 1          # rfft output length
        # Learnable complex filter, one weight per frequency bin
        self.complex_weight = nn.Parameter(
            torch.randn(num_freq_bins, 2, dtype=torch.float32) * 0.02
        )
        self.gate = nn.Parameter(torch.ones(1) * 0.5)  # learnable residual blend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, P, _, C = x.shape
        assert C == self.num_bands

        identity = x
        # Flatten spatial dims so every pixel's band-vector is independent
        x_flat = x.reshape(B * N * P * P, C).to(torch.float32)

        # ── 1-D real FFT along the band axis ──
        X = torch.fft.rfft(x_flat, dim=-1, norm='ortho')   # (B*N*P*P, C//2+1) complex

        # ── Multiply by learnable complex filter ──
        W = torch.view_as_complex(self.complex_weight)      # (C//2+1,) complex
        X = X * W.unsqueeze(0)

        # ── Inverse FFT back to band domain ──
        x_filt = torch.fft.irfft(X, n=C, dim=-1, norm='ortho')
        x_filt = x_filt.reshape(B, N, P, P, C)

        # ── Gated residual ──
        g = torch.sigmoid(self.gate)
        return identity + g * x_filt


# ─────────────────────────────────────────────
# MODULE B: Spatial-Axis Spectral Gating (YOUR EXISTING CODE, RENAMED)
#   2-D FFT over the (a, b) patch-grid axes.
#   Sits INSIDE the transformer stack on tokens.
# ─────────────────────────────────────────────
class SpatialSpectralGatingNetwork(nn.Module):
    """
    Input  : (B, N, D)  – N = a*b tokens on the spatial grid, D = embed dim
    Output : (B, N, D)  – spatial-frequency filtered + residual
    """
    def __init__(self, dim: int, h: int = 14, w: int = 14):
        super().__init__()
        self.h, self.w = h, w
        num_freq_w = w // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(h, num_freq_w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.gate = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        identity = x
        x = x.view(B, a, b, C).to(torch.float32)

        X = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        W = torch.view_as_complex(self.complex_weight)      # (a, b//2+1, C) complex
        X = X * W
        x_filt = torch.fft.irfft2(X, s=(a, b), dim=(1, 2), norm='ortho')
        x_filt = x_filt.reshape(B, N, C)

        g = torch.sigmoid(self.gate)
        return identity + g * x_filt


# ─────────────────────────────────────────────
# MODULE C: HSI Patch Embedding (replaces the
#   simple Linear in the original Predictor)
#   Band gating → flatten → project
# ─────────────────────────────────────────────
class HSIPatchEmbedding(nn.Module):
    """
    Raw patches (B, N, P, P, C)  →  tokens (B, N, D)
    """
    def __init__(self, patch_size: int, num_bands: int, embed_dim: int,
                 use_band_gating: bool = True):
        super().__init__()
        self.use_band_gating = use_band_gating
        if use_band_gating:
            self.band_gate = BandSpectralGatingNetwork(num_bands)
        self.flatten = nn.Flatten(start_dim=2)               # (B,N,P,P,C)→(B,N,P*P*C)
        self.proj    = nn.Linear(patch_size**2 * num_bands, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if self.use_band_gating:
            patches = self.band_gate(patches)                # band-axis FFT filter
        x = self.flatten(patches)                            # (B, N, P*P*C)
        return self.proj(x)                                  # (B, N, D)


# ─────────────────────────────────────────────
# MODULE D: Full Encoder (context or target)
# ─────────────────────────────────────────────
class HSIEncoder(nn.Module):
    def __init__(self, patch_size=16, num_bands=100, embed_dim=128,
                 num_heads=4, mlp_exp=4, depth=8, grid_h=14, grid_w=14,
                 use_band_gating=True, use_spatial_gating=True):
        super().__init__()
        self.patch_embed = HSIPatchEmbedding(
            patch_size, num_bands, embed_dim, use_band_gating
        )
        self.pos_enc = nn.Parameter(torch.zeros(1, grid_h * grid_w, embed_dim))
        self.blocks  = nn.ModuleList([
            ViTBlock(embed_dim, mlp_exp, num_heads) for _ in range(depth)
        ])
        self.spatial_gate = (
            SpatialSpectralGatingNetwork(embed_dim, grid_h, grid_w)
            if use_spatial_gating else None
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, P, P, C) → tokens: (B, N, D)"""
        x = self.patch_embed(patches) + self.pos_enc        # (B, N, D)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # Inject spatial-frequency gating at the midpoint
            if self.spatial_gate is not None and i == len(self.blocks) // 2:
                x = self.spatial_gate(x)
        return self.norm(x)


# ─────────────────────────────────────────────
# MODULE E: Predictor (UPDATED for HSI)
#   Architecture is identical to your original;
#   the only change is that the input tokens it
#   receives already carry band-gated information
#   from the encoder upstream.
# ─────────────────────────────────────────────
class HSIPredictor(nn.Module):
    def __init__(self, block_count=8, embedding_dim=128, attention_heads=4,
                 mlp_expansion=4, patch_count_per_side=14):
        super().__init__()
        self.patch_count_per_side = patch_count_per_side
        self.embedding_dim = embedding_dim

        self.transformer_blocks = nn.ModuleList([
            ViTBlock(embedding_dim, mlp_expansion, attention_heads)
            for _ in range(block_count)
        ])
        self.context_pos_encoding = nn.Parameter(
            torch.zeros(1, patch_count_per_side**2, embedding_dim)
        )
        self.predictor_mask_token = nn.Parameter(
            torch.zeros(1, patch_count_per_side**2, embedding_dim)
        )
        self.output_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, context_representation, context_token_mask, target_block_mask):
        B, N, D = context_representation.shape
        device  = context_representation.device

        # ── Positional encoding ──
        pos = self.context_pos_encoding.expand(B, -1, -1)
        context_representation = context_representation + pos
        mask_tokens = self.predictor_mask_token.expand(B, -1, -1) + pos
        context_representation = torch.cat(
            (context_representation, mask_tokens), dim=1
        )  # (B, 2N, D)

        # ── Build complete token mask ──
        complete_token_mask = torch.zeros(B, N * 2, dtype=torch.bool, device=device)
        for b in range(B):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            for y in range(target_block_mask[b][0], target_block_mask[b][2]):
                for x in range(target_block_mask[b][1], target_block_mask[b][3]):
                    mask[y * self.patch_count_per_side + x] = True
            complete_token_mask[b, :N] = context_token_mask[b]
            complete_token_mask[b, N:] = False

        # ── Transformer blocks ──
        for block in self.transformer_blocks:
            context_representation = block(
                context_representation, complete_token_mask
            )

        return self.output_layer_norm(context_representation[:, N:])


# ─────────────────────────────────────────────
# MODULE F: Full I-JEPA Training Wrapper
# ─────────────────────────────────────────────
class HSIIJEPA(nn.Module):
    def __init__(self, patch_size=16, num_bands=100, embed_dim=128,
                 num_heads=4, mlp_exp=4, enc_depth=8, pred_depth=4,
                 grid_h=14, grid_w=14, ema_decay=0.996,
                 use_band_gating=True, use_spatial_gating=True):
        super().__init__()
        self.ema_decay = ema_decay

        # Two encoders with identical architecture
        self.context_encoder = HSIEncoder(
            patch_size, num_bands, embed_dim, num_heads, mlp_exp,
            enc_depth, grid_h, grid_w,
            use_band_gating=use_band_gating, use_spatial_gating=use_spatial_gating
        )
        self.target_encoder = HSIEncoder(
            patch_size, num_bands, embed_dim, num_heads, mlp_exp,
            enc_depth, grid_h, grid_w,
            use_band_gating=use_band_gating, use_spatial_gating=use_spatial_gating
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        # Freeze target encoder; update via EMA
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = HSIPredictor(
            pred_depth, embed_dim, num_heads, mlp_exp, grid_h
        )

    def get_gradient_descent_parameters(self):
        return list(self.context_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: θ̄ ← τ·θ̄ + (1-τ)·θ"""
        for p_target, p_context in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            p_target.data.mul_(self.ema_decay).add_(
                p_context.data, alpha=1 - self.ema_decay
            )

    def forward(self, patches, context_mask, target_block_mask):
        """
        patches          : (B, N, P, P, C)  raw HSI patches
        context_mask     : (B, S) bool       which tokens are visible
        target_block_mask: (B, 4) int        [y0, x0, y1, x1] of target region
        """
        # 1. Target representations (no gradient)
        with torch.no_grad():
            all_target_tokens = self.target_encoder(patches)  # (B, N, D)

        # 2. Hide target patches before the context encoder sees them.
        context_patches = patches.clone()
        context_patches = context_patches.masked_fill(
            ~context_mask[:, :, None, None, None], 0
        )
        all_context_tokens = self.context_encoder(context_patches)    # (B, N, D)

        # 3. Predict target representations from context
        predicted = self.predictor(
            all_context_tokens, context_mask, target_block_mask
        )  # (B, N, D), target positions only

        target_tokens = []
        predicted_tokens = []
        grid_size = self.predictor.patch_count_per_side
        for batch_index in range(patches.shape[0]):
            y0, x0, y1, x1 = target_block_mask[batch_index].tolist()
            target_grid = all_target_tokens[batch_index].view(grid_size, grid_size, -1)
            predicted_grid = predicted[batch_index].view(grid_size, grid_size, -1)
            target_tokens.append(target_grid[y0:y1, x0:x1].reshape(-1, target_grid.shape[-1]))
            predicted_tokens.append(predicted_grid[y0:y1, x0:x1].reshape(-1, predicted_grid.shape[-1]))
        return torch.stack(predicted_tokens), torch.stack(target_tokens)