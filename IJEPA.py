import torch
import numpy as np
import matplotlib.pyplot as plt

import TargetEncoder
import ContextEncoder
import Predictor

import Patchification
import Visualization

class IJepa(torch.nn.Module):

    def __init__(self):
        super().__init__()

        
        self.M = 4 # Hyperparameter: How many target blocks to sample. In the paper M = 4(M=Target Block)
        
        # ViT architecture - this are the values of DeiT-Tiny
        self.block_count = 12
        self.embedding_dim = 192
        self.mlp_expension = 4
        self.attention_heads = 3
        
        self.patch_size = 16
        self.patch_count_per_side = 14

        self.target_encoder = TargetEncoder.TargetEncoder(block_count = self.block_count, embedding_dim = self.embedding_dim, attention_heads=self.attention_heads, mlp_expension=self.mlp_expension, patch_size = self.patch_size, patch_count_per_side = self.patch_count_per_side)
        self.context_encoder = ContextEncoder.ContextEncoder(block_count = self.block_count, embedding_dim = self.embedding_dim, attention_heads=self.attention_heads, mlp_expension=self.mlp_expension, patch_size = self.patch_size, patch_count_per_side = self.patch_count_per_side)
        self.predictor = Predictor.Predictor(block_count = self.block_count, embedding_dim = self.embedding_dim, attention_heads=self.attention_heads, mlp_expension=self.mlp_expension, patch_size = self.patch_size, patch_count_per_side = self.patch_count_per_side)
        
        # Initialize target encoder weights with context encoder weights
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        
    def get_gradient_descent_parameters(self):
        return list(self.context_encoder.parameters()) + list(self.predictor.parameters())


    def sample_target_blocks(self, x):
        B = x.shape[0]
        
        target_candidates = self.target_encoder(x)

        # For visualizing target blocks, we can replace the target representations with image patches
        #target_candidates = Patchification.patchify(x, patch_size=self.patch_size, as_grid=True)

        # Sample from the target representation
        # The paper specifies that the target blocks are in the range 15% - 20% "scale". That probably means 
        # that a target blocks covers an area of 15% - 20% of the whole image area
        # For a ViT working on 224x224 images with a patch_size of 16x16 pixels that means 29 - 39 patches
        # can be used for the target block
        number_of_target_patches = np.random.randint((int)(self.patch_count_per_side ** 2 * 0.15), (int)(self.patch_count_per_side ** 2 * 0.20))

        # The paper also specifies a aspect ration of 0.75 to 1.5 per target block 
        aspect_ration = np.random.randint(75, 151) / 100

        # Calculate the width and height of the target block in patches
        width_patches = (int)(np.sqrt(number_of_target_patches) * aspect_ration)
        height__patches = (int)(np.sqrt(number_of_target_patches) / aspect_ration)

        # Extract the patches
        target_blocks = []
        target_block_masks = [] # Mask for a given block. In the paper this is specified as "B". Here mask is defined as the coordiante of the top left and bottom right corner

        # Go through every sample of the batch
        for b in range(B):
            target_blocks.append([])
            target_block_masks.append([])

            for m in range(self.M):
                # Randomly select the top-left corner for each batch element
                max_y = (int)(self.patch_count_per_side - height__patches)
                max_x = (int)(self.patch_count_per_side - width_patches)
                y = np.random.randint(0, max_y + 1)  # Random y-coordinate
                x = np.random.randint(0, max_x + 1)  # Random x-coordinate

                patch = target_candidates[b, y:y+height__patches, x:x+width_patches, :]
                
                target_block_masks[-1].append((y, x, y + height__patches, x + width_patches))  # Shape (batch_size, M, 4) 
                target_blocks[-1].append(patch) # (batch_size, M, patch_y, patch_x, embedding_dim)


        # Convert target_blocks to a torch.Tensor
        # Stack patches for each sample in the batch, then stack all samples
        target_blocks = torch.stack([
            torch.stack(sample_patches, dim=0)  # Stack M patches for each sample
            for sample_patches in target_blocks  # Iterate over each sample in the batch
        ], dim=0)  # Stack all samples along the batch dimension

        # Convert target block mask to tensor
        target_block_masks = torch.tensor(target_block_masks)

        # If we replaced the target representations with image patches, we can visualize them here
        #Visualization.visualize_target_blocks(target_blocks, target_block_masks, self.M, self.patch_size)

        flattened_target_blocks = torch.stack([
            Patchification.flatten_patch_grid(target_blocks[:, m, :, :, :]) for m in range(self.M)
        ], dim=1)  # stacking along the M dimension

        return flattened_target_blocks, target_block_masks


    def sample_context_block(self, input, target_block_masks):

        B = input.shape[0]

        scale = np.random.randint(85, 101) / 100
        context_block_patch_width_height = (int)(round(self.patch_count_per_side * scale))
        
        max_x_y = (int)(self.patch_count_per_side - context_block_patch_width_height)
        y = np.random.randint(0, max_x_y + 1)  
        x = np.random.randint(0, max_x_y + 1)  

        context_block = Patchification.patchify(input, patch_size=self.patch_size, as_grid=True)

        # Generate coordinates for the context block patches
        context_block_patch_coords = []
        for b in range(B):
            context_block_patch_coords.append([])
            for i in range(context_block_patch_width_height):
                for k in range(context_block_patch_width_height):
                    context_block_patch_coords[-1].append((y + i, x + k))

        # Mask out areas that are present in the target blocks
        for b in range(B):
            for mask in target_block_masks[b]:
                mask_y_min = mask[0]
                mask_x_min = mask[1]
                mask_y_max = mask[2]
                mask_x_max = mask[3]

                for mask_y in range(mask_y_min, mask_y_max):
                    for mask_x in range(mask_x_min, mask_x_max):
                        if context_block.shape[1] > mask_y and context_block.shape[2] > mask_x:
                            # Mask out patch in content block and delete it from the coordinate list
                            context_block[b, mask_y, mask_x, :] = 0
                        if (mask_y, mask_x) in context_block_patch_coords[b]:
                            context_block_patch_coords[b].remove((mask_y, mask_x))

        # Apply context block scale
        context_block = context_block[:, y:y + context_block_patch_width_height, x:x + context_block_patch_width_height, :]

        flattened_context_block = Patchification.flatten_patch_grid(context_block)
        context_block = Patchification.unpatchify(flattened_context_block, context_block_patch_coords, self.patch_size, self.patch_size * self.patch_count_per_side)
        
        # For visualizing the context and target blocks, we can plot it with matplotlib
        #Visualization.visualize_context_and_target_blocks(input, flattened_context_block, context_block_patch_coords, target_block_masks, self.patch_size, self.patch_count_per_side)    

        return context_block          


    def predict_representation(self, context_representation, context_token_mask, target_block_masks, m):
        
        B = context_representation.shape[0]
        prediction = self.predictor(context_representation, context_token_mask, target_block_masks[:, m, :]) # (Batch, 2 * patch count in image, embedding_dim)

        # Now extract the learnable mask tokens again
        predicted_tokens = prediction[:, self.patch_count_per_side**2:self.patch_count_per_side**2 * 2, :] # (Batch, patch count in image, embedding_dim)
        
        # Get the learnable mask tokens back into a grid structure 
        predicted_tokens = Patchification.griddify_flat_patch(predicted_tokens, patch_size=16) # (Batch, patch_count_per_side, patch_count_per_side, embedding_dim)

        # Now we need to select those that correspond with target_block_masks
        predicted_representations = []
        for b in range(B):
            predicted_representations.append([])

            for y in range(target_block_masks[b, m, 0], target_block_masks[b, m, 2]):
                for x in range(target_block_masks[b, m, 1], target_block_masks[b, m, 3]):
                    predicted_representations[-1].append(predicted_tokens[b, y, x, :]) # Slice out the mask tokens that correspond to the target block

        # predicted_representations have shape (batch, target_block_patch_count, embedding_dim)
        # Convert predicted representations to a torch.Tensor
        predicted_representations = torch.stack([
            torch.stack(sample_patches, dim=0)  # Stack M patches for each sample
            for sample_patches in predicted_representations  # Iterate over each sample in the batch
        ], dim=0)  # Stack all samples along the batch dimension

        return predicted_representations # This should be (Batch, target_block_patch_count, embedding_dim)

    def forward(self, x):

        target_representations, target_block_masks = self.sample_target_blocks(x)
        context_block = self.sample_context_block(x, target_block_masks)
        
        context_representation, context_token_mask = self.context_encoder(context_block)  

        predictions = []
        for m in range(self.M):
            predictions.append(self.predict_representation(context_representation, context_token_mask, target_block_masks, m))

        predictions = torch.stack(predictions, dim=1)  # shape: [batch, M, target_block_patch_count, embedding_dim]

        return target_representations, predictions


       