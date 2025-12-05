import torch
from einops import rearrange

from einops import rearrange

def patchify(imgs, patch_size=16, as_grid=False):
    batch, channels, height, width = imgs.shape
    # Number of patches per side
    h_patches = height // patch_size
    w_patches = width // patch_size
    
    patches = rearrange(imgs, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch_size, pw=patch_size)
    
    if as_grid:
        patches = rearrange(patches, 'b (h w) d -> b h w d', h=h_patches, w=w_patches)
    
    return patches

def flatten_patch_grid(patches_grid):
    batch_size, h_patches, w_patches, patch_vector = patches_grid.shape
    patches_flat = rearrange(patches_grid, 'b h w d -> b (h w) d')
    return patches_flat

def griddify_flat_patch(patches_flat, patch_size, image_size=224):
    batch_size, num_patches, patch_vector = patches_flat.shape
    height_patches = image_size // patch_size
    width_patches = image_size // patch_size

    assert num_patches == height_patches * width_patches, (
        "Number of patches does not match image and patch size")

    patches_grid = rearrange(patches_flat, 'b (h w) d -> b h w d', h=height_patches, w=width_patches)
    return patches_grid


def unpatchify(patches, coords, patch_size=16, image_size=224):

    B = patches.shape[0]
    P = patches.shape[1]

    C = 3  # Assuming RGB

    # Initialize an empty image tensor
    images = torch.zeros(B, C, image_size, image_size, device=patches.device)

    # Iterate over each patch and place it at the correct position
    for b in range(B):
        for p in range(len(coords[b])):
            patch = patches[b, p]
            patch = patch.reshape(C, patch_size, patch_size)
            y, x = coords[b][p]  # Get the (y, x) coordinate for the patch

            # Calculate the pixel coordinates
            y_start = y * patch_size
            y_end = y_start + patch_size
            x_start = x * patch_size
            x_end = x_start + patch_size

            # Place the patch in the image
            images[b, :, y_start:y_end, x_start:x_end] = patch

    return images