import argparse
import csv
import importlib.util
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "ijepa_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

import HoustonLoader


def load_band_module():
    path = Path(__file__).with_name("Band-I-JEPA-Improved.py")
    spec = importlib.util.spec_from_file_location("band_ijepa_improved", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(name):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name.lower(), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def variant_slug(name):
    return name.lower().replace("-", "_")


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_sampler_generator_state(loader):
    generator = getattr(loader.sampler, "generator", None)
    if generator is None:
        return None
    return generator.get_state()


def restore_train_sampler_generator(loader, state):
    generator = getattr(loader.sampler, "generator", None)
    if generator is not None and state is not None:
        generator.set_state(state.cpu())


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


# ═══════════════════════════════════════════════════════════════════════
# FIX 3b: Class-Balanced Sampler for the DataLoader
# ───────────────────────────────────────────────────────────────────────
# Problem: Standard random sampling gives majority classes 95%+ of batches.
# Fix: Inverse frequency sampling — rare classes appear more often per epoch.
# ═══════════════════════════════════════════════════════════════════════
class ClassBalancedSampler(torch.utils.data.Sampler):
    """
    Samples elements according to inverse class frequency.
    Each class has equal probability of being selected per batch position.

    Args:
        labels: LongTensor of class labels for all samples
        num_samples: number of samples to draw per epoch
        beta: controls effective number weighting (0 = uniform, 1 = full inverse)
    """
    def __init__(self, labels: torch.Tensor, num_samples: int,
                 beta: float = 0.9999, num_classes: int = 20):
        super().__init__(labels)
        self.labels = labels
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Compute class frequencies
        unique, counts = torch.unique(labels, return_counts=True)

        # Effective number of samples per class
        effective_num = (1.0 - beta ** counts.float()) / (1.0 - beta + 1e-8)

        # Inverse frequency weight
        per_sample_weights = 1.0 / (effective_num[labels.long()] + 1e-8)

        # Normalize so weights sum to num_classes
        per_sample_weights = per_sample_weights / per_sample_weights.sum() * len(unique)
        self.weights = per_sample_weights

    def __iter__(self):
        # Weighted random sampling with replacement
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


# ═══════════════════════════════════════════════════════════════════════
# FIX 4: Composite Model Selection Metric
# ───────────────────────────────────────────────────────────────────────
# Problem: Checkpointing on raw kNN accuracy → majority class bias.
# Fix: Use balanced accuracy + macro F1 as the selection criterion.
# ═══════════════════════════════════════════════════════════════════════
def composite_score(metrics: dict) -> float:
    """
    Weighted combination of balanced metrics.
    Prioritizes minority class performance while acknowledging overall accuracy.

    Weights chosen to penalize large balanced accuracy drops even if
    raw accuracy improves.
    """
    knn     = metrics.get("knn_accuracy", 0)
    bal_acc = metrics.get("balanced_accuracy", 0)
    macro_f1 = metrics.get("macro_f1", 0)

    # Balanced: 40% raw kNN, 35% balanced accuracy, 25% macro F1
    # The raw kNN still matters because it reflects representation quality.
    return 0.40 * knn + 0.35 * bal_acc + 0.25 * macro_f1


def train_epoch(model, loader, optimizer, device, patch_size,
                class_balanced_loss=None, num_classes=20):
    model.train()
    total_loss = 0.0
    total_count = 0
    grid_size = loader.dataset[0][0].shape[-1] // patch_size

    for windows, labels in loader:
        tokens = to_tokens(windows.to(device, non_blocking=True), patch_size)
        context_mask, target_mask = target_block(grid_size, device)
        batch_size = tokens.shape[0]
        context_mask = context_mask.unsqueeze(0).expand(batch_size, -1)
        target_mask = target_mask.unsqueeze(0).expand(batch_size, -1)

        predicted, target = model(tokens, context_mask, target_mask)

        # ── FIX 3c: Use class-balanced loss ──
        if class_balanced_loss is not None:
            # Get labels for the target patch positions (majority class from batch)
            # In practice: use the label of the first pixel in the target block
            batch_labels = labels.to(device)
            class_balanced_loss.update_class_counts(batch_labels)

            # Reshape for per-sample loss weighting
            B, N, D = predicted.shape
            # Flatten: each target token gets the batch's label as approximation
            pred_flat = predicted.view(batch_size, -1, D)
            tgt_flat = target.view(batch_size, -1, D)
            labels_expanded = batch_labels.unsqueeze(1).expand(-1, pred_flat.shape[1])

            loss = class_balanced_loss(pred_flat, tgt_flat.detach(), labels_expanded)
        else:
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
        n_neighbors=neighbors, weights="distance", algorithm="brute", n_jobs=args.knn_jobs
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return {
        "knn_accuracy": float(accuracy_score(test_labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
    }


def empty_metrics():
    return {"knn_accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan")}


def format_metric(value):
    return "skipped" if np.isnan(value) else f"{value * 100:.2f}%"


def final_score(history, best_composite, best_epoch):
    final = dict(history[-1])
    final["best_composite"] = best_composite
    final["best_epoch"] = best_epoch
    return final


def checkpoint_payload(name, epoch, model, optimizer, metrics, loss, history,
                       best_composite, best_epoch, train_loader):
    return {
        "variant": name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "train_loss": loss,
        "history": history,
        "best_composite": best_composite,
        "best_epoch": best_epoch,
        "rng_state": rng_state(),
        "train_sampler_generator_state": train_sampler_generator_state(train_loader),
    }


def run_variant(name, args, train_dataset, test_dataset, device, use_gates,
                class_balanced_loss=None, num_classes=20):
    set_seed(args.seed)

    # ── FIX 4: Use class-balanced sampler for training ──
    if use_gates:  # Only for Band-I-JEPA
        train_labels_tensor = torch.tensor(
            [train_dataset[i][1] for i in range(len(train_dataset))],
            dtype=torch.long
        )
        sampler = ClassBalancedSampler(
            train_labels_tensor,
            num_samples=len(train_dataset),
            beta=0.9999,
            num_classes=num_classes
        )
        train_loader, test_loader = HoustonLoader.make_loaders(
            train_dataset, test_dataset, args.batch_size, args.num_workers,
            args.seed, sampler=sampler
        )
    else:
        train_loader, test_loader = HoustonLoader.make_loaders(
            train_dataset, test_dataset, args.batch_size, args.num_workers, args.seed
        )

    module = load_band_module() if use_gates else load_module("I-JEPA")
    num_bands = train_dataset.num_bands
    grid_size = train_dataset.patches.shape[-1] // args.patch_size

    model = module.HSIIJEPA(
        patch_size=args.patch_size, num_bands=num_bands, embed_dim=args.embed_dim,
        num_heads=args.num_heads, enc_depth=args.depth, pred_depth=args.depth,
        grid_h=grid_size, grid_w=grid_size,
        use_band_gating=use_gates, use_spatial_gating=use_gates,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.get_gradient_descent_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    best_composite = -1.0
    best_epoch = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = variant_slug(name)
    best_checkpoint_path = args.output_dir / f"{slug}_best.pt"
    last_checkpoint_path = args.output_dir / f"{slug}_last.pt"

    start_epoch = 1
    if args.resume and last_checkpoint_path.exists():
        checkpoint = torch_load(last_checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        best_composite = float(checkpoint.get("best_composite", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint.get("rng_state"))
        restore_train_sampler_generator(train_loader, checkpoint.get("train_sampler_generator_state"))
        if start_epoch > args.epochs:
            print(f"{name} already completed {args.epochs} epochs; using checkpoint.", flush=True)
            return history, final_score(history, best_composite, best_epoch)
        print(f"{name} resumed from epoch {start_epoch - 1:03d}/{args.epochs}.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_epoch(
            model, train_loader, optimizer, device, args.patch_size,
            class_balanced_loss=class_balanced_loss if use_gates else None,
            num_classes=num_classes
        )

        should_evaluate = epoch == args.epochs or epoch % args.knn_every == 0
        metrics = (evaluate_knn(model, train_loader, test_loader, device, args)
                   if should_evaluate else empty_metrics())

        current_composite = composite_score(metrics)

        row = {
            "model": name, "epoch": epoch, "train_loss": loss,
            "composite_score": current_composite, **metrics
        }
        history.append(row)

        print(
            f"{name} epoch {epoch:03d}/{args.epochs} "
            f"loss={loss:.6f} "
            f"knn={format_metric(metrics['knn_accuracy'])} "
            f"bal={format_metric(metrics['balanced_accuracy'])} "
            f"f1={format_metric(metrics['macro_f1'])} "
            f"[composite={current_composite:.4f}]",
            flush=True,
        )

        checkpoint = checkpoint_payload(
            name, epoch, model, optimizer, metrics, loss, history,
            best_composite, best_epoch, train_loader
        )

        # ── FIX 5: Checkpoint on composite score, not raw kNN ──
        if not np.isnan(metrics["knn_accuracy"]) and current_composite > best_composite:
            best_composite = current_composite
            best_epoch = epoch
            checkpoint = checkpoint_payload(
                name, epoch, model, optimizer, metrics, loss, history,
                best_composite, best_epoch, train_loader
            )
            atomic_torch_save(checkpoint, best_checkpoint_path)

        atomic_torch_save(checkpoint, last_checkpoint_path)

        if args.progress_callback is not None:
            args.progress_callback(history)

    return history, final_score(history, best_composite, best_epoch)


def write_csv(path, rows):
    if not rows:
        return
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def make_plots(history, output_dir):
    models = sorted({row["model"] for row in history})

    # Training loss
    plt.figure(figsize=(9, 5))
    for model in models:
        rows = [r for r in history if r["model"] == model]
        plt.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE loss")
    plt.title("Houston18 JEPA training loss (improved)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    # kNN accuracy
    plt.figure(figsize=(9, 5))
    for model in models:
        rows = [r for r in history if r["model"] == model]
        plt.plot([r["epoch"] for r in rows], [r["knn_accuracy"] * 100 for r in rows], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("kNN accuracy (%)")
    plt.title("Houston18 kNN classification accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "knn_accuracy.png", dpi=180)
    plt.close()

    # ── NEW: Balanced accuracy ──
    plt.figure(figsize=(9, 5))
    for model in models:
        rows = [r for r in history if r["model"] == model]
        plt.plot([r["epoch"] for r in rows], [r["balanced_accuracy"] * 100 for r in rows], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Balanced accuracy (%)")
    plt.title("Houston18 Balanced Accuracy (key metric)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "balanced_accuracy.png", dpi=180)
    plt.close()

    # ── NEW: Macro F1 ──
    plt.figure(figsize=(9, 5))
    for model in models:
        rows = [r for r in history if r["model"] == model]
        plt.plot([r["epoch"] for r in rows], [r["macro_f1"] * 100 for r in rows], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1 (%)")
    plt.title("Houston18 Macro F1-Score")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "macro_f1.png", dpi=180)
    plt.close()

    # ── NEW: Composite score ──
    plt.figure(figsize=(9, 5))
    for model in models:
        rows = [r for r in history if r["model"] == model]
        plt.plot([r["epoch"] for r in rows], [r["composite_score"] * 100 for r in rows], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Composite Score (%)")
    plt.title("Houston18 Composite Selection Score (0.40·kNN + 0.35·BalAcc + 0.25·F1)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "composite_score.png", dpi=180)
    plt.close()


def main():
    default_data_dir = Path(os.environ.get("HOUSTON18_DIR", "Houston2018"))

    parser = argparse.ArgumentParser(
        description="Improved comparison: I-JEPA vs Band-I-JEPA on Houston18"
    )
    parser.add_argument("--train-image", type=Path, default=default_data_dir / "HSI_Tr.mat")
    parser.add_argument("--train-label", type=Path, default=default_data_dir / "TrLabel.mat")
    parser.add_argument("--test-image", type=Path, default=default_data_dir / "HSI_Te.mat")
    parser.add_argument("--test-label", type=Path, default=default_data_dir / "TeLabel.mat")
    parser.add_argument("--train-image-key")
    parser.add_argument("--train-label-key")
    parser.add_argument("--test-image-key")
    parser.add_argument("--test-label-key")
    parser.add_argument("--patch-size", type=int, default=1)
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
    parser.add_argument("--knn-every", type=int, default=1)
    parser.add_argument("--knn-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("Results/Houston2018-Improved"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-classes", type=int, default=20,
                        help="Number of land-cover classes (for class-balanced sampling)")
    args = parser.parse_args()

    if args.knn_every < 1:
        raise ValueError("--knn-every must be at least 1")

    set_seed(args.seed)

    train_dataset, test_dataset = HoustonLoader.make_precomputed_datasets(
        args.train_image, args.train_label, args.test_image, args.test_label,
        args.train_image_key, args.train_label_key,
        args.test_image_key, args.test_label_key,
    )

    print("Train samples:", len(train_dataset), "Test samples:", len(test_dataset))
    print("Classes:", args.num_classes)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # ── FIX 3: Class-balanced loss for Band-I-JEPA ──
    class_balanced_loss = ClassBalancedMSELoss(
        num_classes=args.num_classes, beta=0.9999
    )

    history = []
    scores = []

    args.progress_callback = lambda variant_history: write_csv(
        args.output_dir / "training_history.csv", history + variant_history
    )

    # Run I-JEPA (no band gating)
    ijepa_name = "I-JEPA"
    ijepa_history, ijepa_score = run_variant(
        ijepa_name, args, train_dataset, test_dataset, device,
        use_gates=False, class_balanced_loss=None, num_classes=args.num_classes
    )
    history.extend(ijepa_history)
    scores.append(ijepa_score)
    write_csv(args.output_dir / "score_table.csv", scores)

    # Run Band-I-JEPA Improved (with all 4 fixes)
    bandijepa_name = "Band-I-JEPA-Improved"
    bandijepa_history, bandijepa_score = run_variant(
        bandijepa_name, args, train_dataset, test_dataset, device,
        use_gates=True, class_balanced_loss=class_balanced_loss,
        num_classes=args.num_classes
    )
    history.extend(bandijepa_history)
    scores.append(bandijepa_score)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_history.csv", history)
    write_csv(args.output_dir / "score_table.csv", scores)
    make_plots(history, args.output_dir)

    score_json_path = args.output_dir / "score_table.json"
    tmp_json_path = score_json_path.with_name(score_json_path.name + ".tmp")
    tmp_json_path.write_text(json.dumps(scores, indent=2))
    tmp_json_path.replace(score_json_path)

    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()