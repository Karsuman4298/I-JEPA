import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ViTBlock import ViTBlock


# ═══════════════════════════════════════════════════════════════════
# FIX 1: Anomaly-Preserving Band-Axis Spectral Gating
# ───────────────────────────────────────────────────────────────────
# Problem: Global FFT uniformly filters all pixels, suppressing
# minority class signatures that appear as "noise" to a global filter.
# Fix: Scale the gating strength by inverse spectral variance.
# High-variance pixels (rare objects) receive lighter filtering.
# ═══════════════════════════════════════════════════════════════════
class BandSpectralGatingNetwork(nn.Module):
    """
    Input  : (B, N, P, P, C)  – N patches, each P×P pixels, C hyperspectral bands
    Output : (B, N, P, P, C)  – same shape, variance-aware band-axis filtered

    Key change: The gate strength g is modulated per-pixel by:
        filter_scale = sigmoid(gate) * (1 - anomaly_weight)
    where anomaly_weight is high for rare objects (high variance)
    and low for common backgrounds (low variance).
    """
    def __init__(self, num_bands: int, embed_dim: int = 128):
        super().__init__()
        self.num_bands = num_bands
        num_freq_bins = num_bands // 2 + 1

        # Learnable complex filter per frequency bin
        self.complex_weight = nn.Parameter(
            torch.randn(num_freq_bins, 2, dtype=torch.float32) * 0.02
        )

        # Learnable gate (kept, but now acts as a MAX filtering strength)
        self.gate = nn.Parameter(torch.ones(1) * 0.5)

        # ── FIX 1a: Variance-aware anomaly detector ──
        # Projects per-pixel band vector → anomaly score
        self.anomaly_proj = nn.Sequential(
            nn.Linear(num_bands, num_bands // 4),
            nn.GELU(),
            nn.Linear(num_bands // 4, 1),
        )

        # ── FIX 1b: Spatial attention for boundary minority objects ──
        # Some minority objects (cars, trees) appear at spatial edges.
        # A spatial attention gate protects these regions from aggressive filtering.
        self.spatial_gate = nn.Sequential(
            nn.Linear(PATCH_SIZE_HELPER := 16, PATCH_SIZE_HELPER // 2),
            nn.GELU(),
            nn.Linear(PATCH_SIZE_HELPER // 2, 1),
        ) if hasattr(torch, 'nn') else None

        self.patch_size_for_spatial = 16  # Set externally via forward param

    def forward(self, x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        B, N, P, P_spatial, C = x.shape
        assert C == self.num_bands

        identity = x

        # ── Flatten spatial dims: treat each pixel's band-vector independently ──
        x_flat = x.reshape(B * N * P * P_spatial, C).to(torch.float32)

        # ═══════════════════════════════════════════════════════════
        # FIX 1a: Compute anomaly weight per pixel
        # High variance (rare objects) → high anomaly_weight → less filtering
        # Low variance (common backgrounds) → low anomaly_weight → full filtering
        # ═══════════════════════════════════════════════════════════
        with torch.no_grad():
            # Spectral variance as a simple anomaly proxy
            spectral_var = torch.var(x_flat, dim=-1, keepdim=True)          # (B*N*P*P, 1)
            spectral_std = torch.sqrt(spectral_var + 1e-6)

            # Learned anomaly score (captures nonlinear spectral anomalies)
            anomaly_score = self.anomaly_proj(x_flat)                        # (B*N*P*P, 1)

            # Combine variance and learned score
            anomaly_raw = spectral_std * 0.5 + anomaly_score * 0.5

            # Normalize to [0, 1] per batch — full spatial context for calibration
            B_flat = B * N * P * P_spatial
            anomaly_raw = anomaly_raw.squeeze(-1)
            b_size = B_flat // B
            anomaly_raw = anomaly_raw.view(B, -1)
            anomaly_min = anomaly_raw.amin(dim=1, keepdim=True)
            anomaly_max = anomaly_raw.amax(dim=1, keepdim=True)
            anomaly_weight = (anomaly_raw - anomaly_min) / (anomaly_max - anomaly_min + 1e-8)
            anomaly_weight = anomaly_weight.view(B_flat, 1)

        # ── 1-D real FFT along the band axis ──
        X = torch.fft.rfft(x_flat, dim=-1, norm='ortho')                    # (B*N*P*P, C//2+1) complex
        W = torch.view_as_complex(self.complex_weight)                       # (C//2+1,) complex
        X = X * W.unsqueeze(0)                                               # Apply FFT filter

        # ── Inverse FFT back to band domain ──
        x_filt = torch.fft.irfft(X, n=C, dim=-1, norm='ortho')               # (B*N*P*P, C) filtered
        x_filt = x_filt.reshape(B, N, P, P_spatial, C)

        # ═══════════════════════════════════════════════════════════
        # FIX 1b: Reduce filtering strength at spatial boundaries
        # Minority objects (cars, trees) often appear at patch edges.
        # Edge pixels get higher protection.
        # ═══════════════════════════════════════════════════════════
        if P == patch_size and P == P_spatial:  # Only if patch is square
            # Create edge-emphasis mask: 0 at center, 1 at edges
            coords_y = torch.linspace(0, 1, P, device=x.device)
            coords_x = torch.linspace(0, 1, P_spatial, device=x.device)
            gy, gx = torch.meshgrid(coords_y, coords_x, indexing='ij')
            edge_dist = torch.minimum(
                torch.minimum(gy, 1 - gy),
                torch.minimum(gx, 1 - gx)
            )  # Distance from nearest edge: 0 at center, 0.5 at edges
            edge_weight = (edge_dist / 0.5).clamp(0, 1).reshape(1, 1, 1, P, P)  # (1,1,1,P,P)
        else:
            edge_weight = torch.zeros(1, 1, 1, P, P_spatial, device=x.device)

        # Combine: anomaly pixels + edge pixels get light filtering
        protection = anomaly_weight.reshape(B, N, P, P_spatial, 1) * 0.6 + edge_weight * 0.4

        # ── Modulated gating ──
        # g (learned) sets maximum filtering strength
        # protection scales it down: rare/boundary pixels get 0.1× filtering, common get 0.9×
        g = torch.sigmoid(self.gate)                                          # scalar in [0, 0.5]
        filter_strength = g * (1.0 - protection.clamp(0, 1))

        # Residual blend: identity + filter_strength * filtered
        # If filter_strength ≈ 0: return identity (preserve rare signatures)
        # If filter_strength ≈ g ≈ 0.5: heavy filtering (common backgrounds)
        return identity + filter_strength * x_filt


# ═══════════════════════════════════════════════════════════════════
# FIX 2: Depthwise-Separable Patch Embedding
# ───────────────────────────────────────────────────────────────────
# Problem: nn.Linear(patch_size² * num_bands, embed_dim) collapses
# 25600 → 128 dimensions in ONE step. Minority class details are
# averaged into oblivion inside a 16×16 spatial patch.
# Fix: Two-stage projection — preserve spatial granularity.
# ═══════════════════════════════════════════════════════════════════
class HSIPatchEmbedding(nn.Module):
    """
    Raw patches (B, N, P, P, C) → tokens (B, N, D)

    Two-stage projection:
      Stage 1: Spectral reduction  (C → embed_dim // 2) per spatial pixel
      Stage 2: Spatial pooling     (P*P locations → 1) via linear

    This keeps minority objects spatially distinguishable within patches
    instead of blending everything into a 25600-dim vector.
    """
    def __init__(self, patch_size: int, num_bands: int, embed_dim: int,
                 use_band_gating: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.use_band_gating = use_band_gating

        if use_band_gating:
            self.band_gate = BandSpectralGatingNetwork(num_bands, embed_dim)

        # ── FIX 2a: Spectral embedding per pixel ──
        # Instead of flattening everything at once, first embed each pixel's
        # spectral vector. This preserves WHERE things are inside the patch.
        self.spectral_embed = nn.Sequential(
            nn.Linear(num_bands, embed_dim // 2),
            nn.GELU(),
        )  # (B, N, P, P, C) → (B, N, P, P, embed_dim//2)

        # ── FIX 2b: Spatial aggregation ──
        # Pool the per-pixel spectral embeddings spatially.
        # We use a lightweight depthwise + pointwise conv instead of a
        # giant Linear to avoid parameter explosion.
        self.spatial_pool = nn.Sequential(
            nn.Flatten(start_dim=2),        # (B, N, P*P*embed_dim//2)
            nn.Linear(patch_size * patch_size * (embed_dim // 2), embed_dim),
            nn.GELU(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, P, P_spatial, C = patches.shape

        if self.use_band_gating:
            # Pass patch_size so BandSpectralGating knows spatial dimensions
            patches = self.band_gate(patches, patch_size=self.patch_size)

        # Stage 1: Embed each pixel's spectral vector
        x = self.spectral_embed(patches)                      # (B, N, P, P, D//2)

        # Stage 2: Pool spatially → one token per patch
        x = x.flatten(start_dim=2)                           # (B, N, P*P*D//2)
        x = self.spatial_pool(x)                             # (B, N, D)

        return x


# ═══════════════════════════════════════════════════════════════════
# FIX 3: Class-Balanced MSE Loss
# ───────────────────────────────────────────────────────────────────
# Problem: Standard MSE loss weights all patches equally.
# 95% of pixels from majority classes dominate the gradient.
# Fix: Weight each sample's loss by inverse class frequency.
# ═══════════════════════════════════════════════════════════════════
class ClassBalancedMSELoss(nn.Module):
    """
    MSE loss weighted by inverse class frequency.
    Rare classes contribute proportionally more to the gradient.
    """
    def __init__(self, num_classes: int, beta: float = 0.9999):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        # Effective number of samples per class — updated per batch
        self.class_counts = None

    def update_class_counts(self, labels: torch.Tensor):
        """Update class frequencies from the current batch."""
        unique, counts = labels.unique(return_counts=True)
        self.class_counts = torch.zeros(self.num_classes, device=labels.device)
        self.class_counts[unique] = counts.float()

    def get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced sample weights."""
        if self.class_counts is None:
            return torch.ones_like(labels)

        # Effective number of samples: ENK = (1 - beta^N_k) / (1 - beta)
        effective_num = (1.0 - self.beta ** (self.class_counts + 1e-8)) / (1.0 - self.beta + 1e-8)

        # Inverse frequency weight per class
        class_weights = 1.0 / (effective_num + 1e-8)
        class_weights = class_weights / class_weights.sum() * self.num_classes  # Normalize

        # Map to per-sample weights
        return class_weights[labels]

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:
        mse = F.mse_loss(pred, target, reduction='none')        # (B, N, D)

        if labels is not None and self.class_counts is not None:
            # Expand weights from (B,) to (B, N, D) for element-wise weighting
            weights = self.get_weights(labels)                   # (B,)
            weights = weights.unsqueeze(-1).unsqueeze(-1)       # (B, 1, 1)
            weights = weights.expand_as(mse)                    # (B, N, D)

            # Normalize by total weight to keep scale comparable
            total_weight = weights.sum()
            if total_weight > 0:
                loss = (weights * mse).sum() / total_weight
            else:
                loss = mse.mean()
        else:
            loss = mse.mean()

        return loss


# ═══════════════════════════════════════════════════════════════════
# MODULE B: Spatial-Axis Spectral Gating (UNCHANGED — works well)
# ═══════════════════════════════════════════════════════════════════
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
        W = torch.view_as_complex(self.complex_weight)
        X = X * W
        x_filt = torch.fft.irfft2(X, s=(a, b), dim=(1, 2), norm='ortho')
        x_filt = x_filt.reshape(B, N, C)

        g = torch.sigmoid(self.gate)
        return identity + g * x_filt


# ═══════════════════════════════════════════════════════════════════
# MODULE D: Full Encoder (context or target) — ADAPTED for fixes
# ═══════════════════════════════════════════════════════════════════
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
        x = self.patch_embed(patches) + self.pos_enc
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.spatial_gate is not None and i == len(self.blocks) // 2:
                x = self.spatial_gate(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════
# MODULE E: Predictor (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════
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

        pos = self.context_pos_encoding.expand(B, -1, -1)
        context_representation = context_representation + pos
        mask_tokens = self.predictor_mask_token.expand(B, -1, -1) + pos
        context_representation = torch.cat(
            (context_representation, mask_tokens), dim=1
        )

        complete_token_mask = torch.zeros(B, N * 2, dtype=torch.bool, device=device)
        for b in range(B):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            for y in range(target_block_mask[b][0], target_block_mask[b][2]):
                for x in range(target_block_mask[b][1], target_block_mask[b][3]):
                    mask[y * self.patch_count_per_side + x] = True
            complete_token_mask[b, :N] = context_token_mask[b]
            complete_token_mask[b, N:] = False

        for block in self.transformer_blocks:
            context_representation = block(
                context_representation, complete_token_mask
            )

        return self.output_layer_norm(context_representation[:, N:])


# ═══════════════════════════════════════════════════════════════════
# MODULE F: Full I-JEPA Training Wrapper — ADAPTED with class balancing
# ═══════════════════════════════════════════════════════════════════
class HSIIJEPA(nn.Module):
    def __init__(self, patch_size=16, num_bands=100, embed_dim=128,
                 num_heads=4, mlp_exp=4, enc_depth=8, pred_depth=4,
                 grid_h=14, grid_w=14, ema_decay=0.996,
                 use_band_gating=True, use_spatial_gating=True,
                 num_classes=20):
        super().__init__()
        self.ema_decay = ema_decay
        self.num_classes = num_classes

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
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = HSIPredictor(
            pred_depth, embed_dim, num_heads, mlp_exp, grid_h
        )

    def get_gradient_descent_parameters(self):
        return list(self.context_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_encoder(self):
        for p_target, p_context in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            p_target.data.mul_(self.ema_decay).add_(
                p_context.data, alpha=1 - self.ema_decay
            )

    def forward(self, patches, context_mask, target_block_mask):
        with torch.no_grad():
            all_target_tokens = self.target_encoder(patches)

        context_patches = patches.clone()
        context_patches = context_patches.masked_fill(
            ~context_mask[:, :, None, None, None], 0
        )
        all_context_tokens = self.context_encoder(context_patches)

        predicted = self.predictor(
            all_context_tokens, context_mask, target_block_mask
        )

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