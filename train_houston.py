import argparse
import csv
import importlib.util
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

import HoustonLoader


def load_band_module():
    path = Path(__file__).with_name("Band-I-JEPA.py")
    spec = importlib.util.spec_from_file_location("band_ijepa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tokens(windows, patch_size):
    patches = windows.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous()
    return patches.view(patches.shape[0], -1, patch_size, patch_size, windows.shape[1])


def target_block(grid_size, device):
    size = max(2, grid_size // 4)
    start = (grid_size - size) // 2
    context_mask = torch.ones(grid_size, grid_size, dtype=torch.bool, device=device)
    context_mask[start:start + size, start:start + size] = False
    block = torch.tensor([start, start, start + size, start + size], device=device)
    return context_mask.flatten(), block


def train_epoch(model, loader, optimizer, device, patch_size):
    model.train()
    total_loss = 0.0
    total_count = 0
    grid_size = 224 // patch_size
    for windows, _ in loader:
        tokens = to_tokens(windows.to(device, non_blocking=True), patch_size)
        context_mask, target_mask = target_block(grid_size, device)
        batch_size = tokens.shape[0]
        context_mask = context_mask.unsqueeze(0).expand(batch_size, -1)
        target_mask = target_mask.unsqueeze(0).expand(batch_size, -1)
        predicted, target = model(tokens, context_mask, target_mask)
        loss = torch.nn.functional.mse_loss(predicted, target.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_gradient_descent_parameters(), 1.0)
        optimizer.step()
        model.update_target_encoder()
        total_loss += loss.item() * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def extract_features(model, loader, device, patch_size):
    model.eval()
    features, labels = [], []
    for windows, batch_labels in loader:
        tokens = to_tokens(windows.to(device, non_blocking=True), patch_size)
        encoded = model.context_encoder(tokens)
        features.append(torch.nn.functional.normalize(encoded.mean(dim=1), dim=1).cpu())
        labels.append(batch_labels)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def evaluate_knn(model, train_loader, test_loader, device, args):
    train_features, train_labels = extract_features(model, train_loader, device, args.patch_size)
    test_features, test_labels = extract_features(model, test_loader, device, args.patch_size)
    neighbors = min(args.k, len(train_labels))
    classifier = KNeighborsClassifier(
        n_neighbors=neighbors, weights="distance", algorithm="brute", n_jobs=-1
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return {
        "knn_accuracy": float(accuracy_score(test_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
    }


def run_variant(name, args, train_dataset, test_dataset, device, use_gates):
    set_seed(args.seed)
    train_loader, test_loader = HoustonLoader.make_loaders(
        train_dataset, test_dataset, args.batch_size, args.num_workers, args.seed
    )
    module = load_band_module()
    num_bands = train_dataset.cube.shape[0]
    grid_size = args.window_size // args.patch_size
    model = module.HSIIJEPA(
        patch_size=args.patch_size, num_bands=num_bands, embed_dim=args.embed_dim,
        num_heads=args.num_heads, enc_depth=args.depth, pred_depth=args.depth,
        grid_h=grid_size, grid_w=grid_size,
        use_band_gating=use_gates, use_spatial_gating=use_gates,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.get_gradient_descent_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    best_accuracy = -1.0
    best_epoch = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, args.patch_size)
        metrics = evaluate_knn(model, train_loader, test_loader, device, args)
        row = {"model": name, "epoch": epoch, "train_loss": loss, **metrics}
        history.append(row)
        print(
            f"{name} epoch {epoch:03d}/{args.epochs} "
            f"loss={loss:.6f} knn={metrics['knn_accuracy'] * 100:.2f}%"
        )
        if metrics["knn_accuracy"] > best_accuracy:
            best_accuracy = metrics["knn_accuracy"]
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics},
                args.output_dir / f"{name.lower().replace('-', '_')}_best.pt",
            )
    final = dict(history[-1])
    final["best_knn_accuracy"] = best_accuracy
    final["best_epoch"] = best_epoch
    return history, final


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(history, output_dir):
    models = sorted({row["model"] for row in history})
    plt.figure(figsize=(9, 5))
    for model in models:
        rows = [row for row in history if row["model"] == model]
        plt.plot([row["epoch"] for row in rows], [row["train_loss"] for row in rows], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE loss")
    plt.title("Houston18 JEPA training loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for model in models:
        rows = [row for row in history if row["model"] == model]
        plt.plot([row["epoch"] for row in rows], [row["knn_accuracy"] * 100 for row in rows], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("kNN accuracy (%)")
    plt.title("Houston18 kNN classification accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "knn_accuracy.png", dpi=180)
    plt.close()


def main():
    default_data_dir = Path(os.environ.get("HOUSTON18_DIR", "Houston2018"))
    parser = argparse.ArgumentParser(description="Compare I-JEPA and Band-I-JEPA on Houston18")
    parser.add_argument("--image", type=Path, default=default_data_dir / "Houston18.mat")
    parser.add_argument("--labels", type=Path, default=default_data_dir / "Houston18_7gt.mat")
    parser.add_argument("--image-key", default="ori_data")
    parser.add_argument("--label-key", default="map")
    parser.add_argument("--window-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("Results/Houston2018"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.window_size % args.patch_size or args.window_size != 224:
        raise ValueError("window-size must currently be 224 and divisible by patch-size")
    set_seed(args.seed)
    train_dataset, test_dataset, labels = HoustonLoader.make_split_datasets(
        args.image, args.labels, args.image_key, args.label_key,
        args.window_size, args.train_fraction, args.seed,
    )
    print("Houston18 cube labels:", dict(zip(*np.unique(labels[labels > 0], return_counts=True))))
    print("Train samples:", len(train_dataset), "Test samples:", len(test_dataset))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    history = []
    scores = []
    for name, use_gates in (("I-JEPA", False), ("Band-I-JEPA", True)):
        variant_history, final_score = run_variant(
            name, args, train_dataset, test_dataset, device, use_gates
        )
        history.extend(variant_history)
        scores.append(final_score)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_history.csv", history)
    write_csv(args.output_dir / "score_table.csv", scores)
    make_plots(history, args.output_dir)
    (args.output_dir / "score_table.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
