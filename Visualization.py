import matplotlib.pyplot as plt
import Patchification

def visualize_context_and_target_blocks(input, flattened_context_block, context_block_patch_coords, target_block_masks, patch_size, patch_count_per_side):
    
    img = Patchification.unpatchify(flattened_context_block, context_block_patch_coords, patch_size, patch_size * patch_count_per_side)
    img = img.cpu()[0].permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()

    patchified = Patchification.patchify(input)

    # For visualization only 
    for b in range(input.shape[0]):
        for mask in target_block_masks[b]:
            
            mask_y_min = mask[0]
            mask_x_min = mask[1]
            mask_y_max = mask[2]
            mask_x_max = mask[3]
    
            img = patchified[b, mask_y_min:mask_y_max, mask_x_min:mask_x_max, :].unsqueeze(0)
        
            coords = []
            coords.append([])
            for i in range(mask_y_min, mask_y_max):
                for k in range(mask_x_min, mask_x_max):
                    coords[-1].append((i, k))

            flattened_imgs = img.reshape(1, -1, 768)

            unpatchified = Patchification.unpatchify(flattened_imgs, coords)

            unpatchified = unpatchified.cpu()[0].permute(1, 2, 0).numpy()
            plt.imshow(unpatchified)
            plt.show()

def visualize_target_blocks(target_blocks, block_mask, M, patch_size):

    B = target_blocks.shape[0]

    for b in range(B):
        for m in range(M):

            coords = []
            coords.append([])
            for y in range(block_mask[b][m][0], block_mask[b][m][2]):
                for x in range(block_mask[b][m][1], block_mask[b][m][3]):
                    coords[-1].append((y, x))

            img = Patchification.unpatchify(target_blocks[b][m].reshape(1, len(coords[0]), patch_size**2*3), coords)
            img = img.cpu()[0].permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.show()