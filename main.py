import torch
import ImageNetLoader

import KnnMonitor
from IJEPA import IJepa  

def main():

    num_epochs = 300
    start_epoch = 0
    warmup_epochs = 10  
    initial_lr = 0.003

    ema_initial_momentum = 0.996 

    initial_weight_decay = 0.04
    final_weight_decay = 0.4

    # For limited compute reasons miniImageNet is used here (100 classes, 600 images per class)
    train_loader, val_loader, _ = ImageNetLoader.get_mini_imagenet_loaders()

    img, _ = next(iter(train_loader))

    ijepa = IJepa().cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(ijepa.get_gradient_descent_parameters(), lr=initial_lr, weight_decay=initial_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=10e-6)
    kNN_monitor = KnnMonitor.KnnMonitor()

    # -----------------
    # Training Loop as specified in the paper with first 10 epochs of warmup, linearly increased weight decay 
    # and EMA momentum
    # -----------------
    for epoch in range(start_epoch, num_epochs):

        running_loss = 0
        num_seen = 0

        ijepa.train()
        for step, (img, _) in enumerate(train_loader, start=1):
            
            # Move image to GPU
            img = img.cuda()

            # Warmup logic
            if epoch < warmup_epochs:
                warmup_lr = initial_lr * (epoch * len(train_loader) + step) / (warmup_epochs * len(train_loader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Linearly interpolate the weight decay for the current epoch
            current_wd = initial_weight_decay + (final_weight_decay - initial_weight_decay) * ((epoch * len(train_loader) + step) / (num_epochs * len(train_loader)))
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = current_wd

            optimizer.zero_grad()
            
            # Get target and predicted representations for an image
            target_representations, predicted_representations = ijepa(img)

            # Calculate MSE Loss between representations
            loss = criterion(predicted_representations, target_representations)
            loss.backward() 

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(ijepa.get_gradient_descent_parameters(), max_norm=1.0)

            optimizer.step()

            # Update target encoder weights with an exponential moving average
            with torch.no_grad():
                momentum = ema_initial_momentum + ((1 - ema_initial_momentum) / (num_epochs - 1)) * epoch
                for param_context, param_target in zip(ijepa.context_encoder.parameters(), ijepa.target_encoder.parameters()):
                    param_target.data.mul_(momentum).add_(param_context.data * (1.0 - momentum))

            # Print loss 
            bs = len(img)
            running_loss += loss.item() * bs
            num_seen += bs
            avg_loss = running_loss / max(1, num_seen)
            pct = 100.0 * num_seen / len(train_loader.dataset)

            print(f"Epoch {epoch} ({pct:.1f}%) Loss: {avg_loss:.4f}", end="\r")

        # Update learning rate
        if epoch >= warmup_epochs:
            scheduler.step()

        val_acc = kNN_monitor.knn_evaluate(train_loader, val_loader, ijepa, k=20)
        print(f"\nEpoch {epoch} kNN Validation Accuracy: {val_acc * 100:.2f}%")
        kNN_monitor.save_accuracy_plot()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ijepa.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, 'checkpoint.pth')

main()