
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Small helper class that trains a kNN classifier on the output features of the encoder
# and than tests the accuracy with the validation dataset
class KnnMonitor:

    def __init__(self):
        self.accuracies = []

    def _extract_features(self, dataloader, model):
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in dataloader:
                imgs = imgs.cuda()
                feats, _ = model.context_encoder(imgs)  
                feats = feats.mean(dim=1)  # shape: [batch_size, embed_dim]
                feats = torch.nn.functional.normalize(feats, dim=1)  
                features.append(feats.cpu())
                labels.append(lbls)
        return torch.cat(features), torch.cat(labels)

    def knn_evaluate(self, train_loader, val_loader, model, k=20):
        train_feats, train_labels = self._extract_features(train_loader, model)
        val_feats, val_labels = self._extract_features(val_loader, model)
        
        # Convert to NumPy (ensure on CPU)
        train_feats_np = train_feats.cpu().numpy()
        train_labels_np = train_labels.cpu().numpy()
        val_feats_np   = val_feats.cpu().numpy()
        val_labels_np  = val_labels.cpu().numpy()


        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(train_feats_np, train_labels_np)

        val_preds = knn.predict(val_feats_np)
        accuracy = (val_preds == val_labels_np).mean()
        self.accuracies.append(accuracy)
        return accuracy

    def save_accuracy_plot(self):
        epochs = np.arange(1, (len(self.accuracies) + 1) * 1, 1)  
        accuracies_percent = np.array(self.accuracies) * 100  

        plt.plot(epochs, accuracies_percent, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("kNN Classification Accuracy Over Epochs")
        plt.grid(True)
        plt.savefig("kNN_classification_accuracy.png")