import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from train_underwater_classifier_v1 import (
    build_transforms,
    build_weighted_sampler,
    class_weights_from_trainset,
    load_hard_negative_keywords,
)


LABELS = ["UXO", "non_UXO"]


def resize_aspect_with_padding(img, target=224, pad_color=(0, 0, 0)):
    w, h = img.size
    if w <= 0 or h <= 0:
        return img.resize((target, target), Image.Resampling.BILINEAR)
    scale = float(target) / float(max(w, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target, target), pad_color)
    canvas.paste(img, ((target - nw) // 2, (target - nh) // 2))
    return canvas


class ManifestImageDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform
        self.classes = LABELS
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = [(row["file"], self.class_to_idx[row["label"]]) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        label = self.class_to_idx[row["label"]]
        with Image.open(row["file"]) as im:
            img = resize_aspect_with_padding(im.convert("RGB"), target=224)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_manifest_rows(manifest_path):
    with manifest_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def assign_groups_to_folds(group_count_map, num_folds, rng):
    items = list(group_count_map.items())
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)

    fold_groups = [set() for _ in range(num_folds)]
    fold_counts = [0 for _ in range(num_folds)]

    for group_id, count in items:
        fold_idx = min(range(num_folds), key=lambda idx: (fold_counts[idx], len(fold_groups[idx])))
        fold_groups[fold_idx].add(group_id)
        fold_counts[fold_idx] += count

    return fold_groups, fold_counts


def build_outer_group_folds(rows, num_folds, seed):
    rng = random.Random(seed)
    by_label_group = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_label_group[row["label"]][row["group_id"]] += 1

    fold_group_sets = [set() for _ in range(num_folds)]
    fold_stats = {label: [] for label in LABELS}

    for label in LABELS:
        groups, counts = assign_groups_to_folds(by_label_group[label], num_folds, rng)
        fold_stats[label] = counts
        for idx in range(num_folds):
            fold_group_sets[idx].update(groups[idx])

    return fold_group_sets, fold_stats


def split_train_val_groups(rows, candidate_groups, val_ratio, seed):
    rng = random.Random(seed)
    by_label_group = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row["group_id"] in candidate_groups:
            by_label_group[row["label"]][row["group_id"]] += 1

    train_groups = set()
    val_groups = set()

    for label in LABELS:
        items = list(by_label_group[label].items())
        rng.shuffle(items)
        items.sort(key=lambda x: x[1], reverse=True)

        if len(items) <= 1:
            for group_id, _ in items:
                train_groups.add(group_id)
            continue

        total = sum(count for _, count in items)
        target_val = max(1, int(round(total * val_ratio)))
        val_count = 0

        for idx, (group_id, count) in enumerate(items):
            remaining = len(items) - idx - 1
            # Keep at least one group for train.
            if val_count < target_val and remaining >= 1:
                val_groups.add(group_id)
                val_count += count
            else:
                train_groups.add(group_id)

        # Safety fallback if greedy selection took all groups.
        if not train_groups.intersection(by_label_group[label].keys()):
            moved_group = next(iter(val_groups.intersection(by_label_group[label].keys())))
            val_groups.remove(moved_group)
            train_groups.add(moved_group)

    return train_groups, val_groups


def select_rows_by_groups(rows, allowed_groups):
    return [row for row in rows if row["group_id"] in allowed_groups]


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    acc = correct / max(total, 1)
    return acc, all_labels, all_preds


def mean_std(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run group-aware K-fold CV for the underwater UXO project.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\manifests\underwater_dataset_v1_manifest.csv"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--augmentation-profile",
        type=str,
        default="underwater",
        choices=["basic", "underwater"],
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use weighted random sampler on the training fold.",
    )
    parser.add_argument(
        "--hard-negative-keywords",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\manifests\hard_negative_keywords_conservative.txt"),
    )
    parser.add_argument("--hard-negative-factor", type=float, default=1.2)
    parser.add_argument(
        "--fold-model-dir",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\models\groupcv_underwater_hn12_cons"),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\metrics\underwater_groupcv_3fold_underwater_hn12_cons.json"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rows = load_manifest_rows(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hard_negative_keywords = load_hard_negative_keywords(args.hard_negative_keywords)

    print(f"Device: {device}")
    print(f"Manifest: {args.manifest}")
    print(f"Rows: {len(rows)}")
    print(f"Num folds: {args.num_folds}")
    print(f"Augmentation profile: {args.augmentation_profile}")
    print(f"Hard negative keywords: {hard_negative_keywords}")
    print(f"Hard negative factor: {args.hard_negative_factor}")

    fold_group_sets, fold_stats = build_outer_group_folds(rows, args.num_folds, seed=args.seed)
    print(f"Outer fold group stats: {fold_stats}")

    train_tf, eval_tf = build_transforms(augmentation_profile=args.augmentation_profile)
    args.fold_model_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for fold_idx in range(args.num_folds):
        test_groups = fold_group_sets[fold_idx]
        train_candidate_groups = set().union(*[fold_group_sets[i] for i in range(args.num_folds) if i != fold_idx])
        train_groups, val_groups = split_train_val_groups(
            rows,
            candidate_groups=train_candidate_groups,
            val_ratio=args.val_ratio,
            seed=args.seed + fold_idx,
        )

        train_rows = select_rows_by_groups(rows, train_groups)
        val_rows = select_rows_by_groups(rows, val_groups)
        test_rows = select_rows_by_groups(rows, test_groups)

        train_ds = ManifestImageDataset(train_rows, transform=train_tf)
        val_ds = ManifestImageDataset(val_rows, transform=eval_tf)
        test_ds = ManifestImageDataset(test_rows, transform=eval_tf)

        class_weights, class_counts = class_weights_from_trainset(train_ds)
        class_weights = class_weights.to(device)

        print(
            f"\nFold {fold_idx + 1}/{args.num_folds}: "
            f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
            f"class_counts={dict(zip(train_ds.classes, class_counts))}"
        )

        if args.balanced_sampling:
            sampler, hard_negative_hits = build_weighted_sampler(
                train_ds,
                class_weights.cpu(),
                hard_negative_keywords=hard_negative_keywords,
                hard_negative_factor=args.hard_negative_factor,
            )
            print(f"Hard negative matches in train fold: {hard_negative_hits}")
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=args.num_workers,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )

        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_acc = -1.0
        history = []
        fold_model_path = args.fold_model_dir / f"fold{fold_idx + 1}_best.pth"

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(len(train_loader), 1)
            val_acc, _, _ = evaluate(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": avg_loss, "val_acc": val_acc})
            print(f"Fold {fold_idx + 1} | Epoch {epoch:02d} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_model_path)

        state = torch.load(fold_model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        val_acc, val_labels, val_preds = evaluate(model, val_loader, device)
        test_acc, test_labels, test_preds = evaluate(model, test_loader, device)

        val_report = classification_report(
            val_labels,
            val_preds,
            target_names=val_ds.classes,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        test_report = classification_report(
            test_labels,
            test_preds,
            target_names=test_ds.classes,
            digits=4,
            output_dict=True,
            zero_division=0,
        )

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "best_val_acc": best_val_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "test_size": len(test_ds),
                "train_class_counts": dict(zip(train_ds.classes, class_counts)),
                "val_confusion_matrix": confusion_matrix(val_labels, val_preds).tolist(),
                "test_confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
                "val_report": val_report,
                "test_report": test_report,
                "history": history,
                "model_path": str(fold_model_path),
            }
        )

    summary = {
        "config": {
            "seed": args.seed,
            "num_folds": args.num_folds,
            "val_ratio": args.val_ratio,
            "augmentation_profile": args.augmentation_profile,
            "balanced_sampling": args.balanced_sampling,
            "hard_negative_keywords": hard_negative_keywords,
            "hard_negative_factor": args.hard_negative_factor,
            "manifest": str(args.manifest),
        },
        "fold_results": fold_results,
        "aggregate": {
            "best_val_acc": mean_std([x["best_val_acc"] for x in fold_results]),
            "val_acc": mean_std([x["val_acc"] for x in fold_results]),
            "test_acc": mean_std([x["test_acc"] for x in fold_results]),
            "test_uxo_recall": mean_std([x["test_report"]["UXO"]["recall"] for x in fold_results]),
            "test_uxo_f1": mean_std([x["test_report"]["UXO"]["f1-score"] for x in fold_results]),
            "test_non_uxo_recall": mean_std([x["test_report"]["non_UXO"]["recall"] for x in fold_results]),
            "test_macro_f1": mean_std([x["test_report"]["macro avg"]["f1-score"] for x in fold_results]),
        },
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Aggregate summary ===")
    for key, value in summary["aggregate"].items():
        print(f"{key}: mean={value['mean']:.4f}, std={value['std']:.4f}")
    print(f"Saved summary to: {args.metrics_out}")


if __name__ == "__main__":
    main()
