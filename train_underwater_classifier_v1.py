import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


class RandomUnderwaterDegradation:
    """Simulate mild turbidity, blur, and contrast loss typical of underwater optics."""

    def __init__(self, p=0.55):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        out = img

        if random.random() < 0.7:
            radius = random.uniform(0.4, 1.4)
            out = out.filter(ImageFilter.GaussianBlur(radius=radius))

        if random.random() < 0.9:
            contrast = random.uniform(0.55, 0.9)
            brightness = random.uniform(0.85, 1.12)
            saturation = random.uniform(0.8, 1.08)
            out = ImageEnhance.Contrast(out).enhance(contrast)
            out = ImageEnhance.Brightness(out).enhance(brightness)
            out = ImageEnhance.Color(out).enhance(saturation)

        if random.random() < 0.65:
            haze_strength = random.uniform(0.08, 0.22)
            haze_color = np.array(
                [
                    random.randint(150, 185),
                    random.randint(165, 205),
                    random.randint(165, 215),
                ],
                dtype=np.float32,
            )
            arr = np.asarray(out).astype(np.float32)
            arr = (1.0 - haze_strength) * arr + haze_strength * haze_color
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            out = transforms.functional.to_pil_image(arr)

        return out


def resize_aspect_with_padding(img, target=224, pad_color=(0, 0, 0)):
    """Isotropic resize + centered padding so mixed-source crops stack cleanly."""
    resampling = getattr(Image, "Resampling", Image)
    bilinear = getattr(resampling, "BILINEAR")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img.resize((target, target), bilinear)

    scale = float(target) / float(max(w, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), bilinear)
    canvas = Image.new("RGB", (target, target), pad_color)
    canvas.paste(img, ((target - nw) // 2, (target - nh) // 2))
    return canvas


def build_transforms(augmentation_profile="underwater"):
    if augmentation_profile == "basic":
        train_steps = [
            transforms.Lambda(lambda img: resize_aspect_with_padding(img, target=224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    else:
        train_steps = [
            transforms.Lambda(lambda img: resize_aspect_with_padding(img, target=224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=18),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.06, 0.06),
                scale=(0.9, 1.08),
                shear=(-6, 6),
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.3, saturation=0.15, hue=0.03),
            RandomUnderwaterDegradation(p=0.65),
        ]

    train_tf = transforms.Compose(
        train_steps
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Lambda(lambda img: resize_aspect_with_padding(img, target=224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


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


def class_weights_from_trainset(train_ds):
    counts = [0 for _ in train_ds.classes]
    for _, label in train_ds.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (len(counts) * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32), counts


def build_train_loader(
    train_ds,
    batch_size,
    num_workers,
    balanced_sampling,
    class_weights,
    hard_negative_keywords=None,
    hard_negative_factor=1.0,
):
    if balanced_sampling:
        sampler, hard_negative_hits = build_weighted_sampler(
            train_ds,
            class_weights,
            hard_negative_keywords=hard_negative_keywords,
            hard_negative_factor=hard_negative_factor,
        )
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
        )
        return loader, hard_negative_hits

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return loader, 0


def run_train_epochs(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epochs,
    stage_name,
):
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        history.append({"epoch": epoch, "train_loss": avg_loss})
        print(f"{stage_name} epoch {epoch:02d} | train_loss={avg_loss:.4f}")
    return history


def load_hard_negative_keywords(path):
    if path is None or not path.exists():
        return []
    keywords = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keywords.append(line.lower())
    return keywords


def build_weighted_sampler(
    train_ds,
    class_weights,
    hard_negative_keywords=None,
    hard_negative_factor=1.0,
):
    sample_weights = []
    hard_negative_hits = 0
    non_uxo_index = train_ds.class_to_idx.get("non_UXO")
    keywords = hard_negative_keywords or []

    for path_str, label in train_ds.samples:
        weight = float(class_weights[label])
        lowered = path_str.lower()
        if (
            non_uxo_index is not None
            and label == non_uxo_index
            and keywords
            and any(keyword in lowered for keyword in keywords)
        ):
            weight *= hard_negative_factor
            hard_negative_hits += 1
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, hard_negative_hits


def main():
    parser = argparse.ArgumentParser(description="Train underwater UXO v1 classifier.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--aux-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aux-data-root",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\aux_recordings_optical_v1"),
        help="Optional auxiliary optical dataset used before TU fine-tuning.",
    )
    parser.add_argument(
        "--augmentation-profile",
        type=str,
        default="underwater",
        choices=["basic", "underwater"],
        help="Training augmentation profile. Use 'underwater' for stronger optical underwater augmentation.",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use weighted random sampler on training set for class balance.",
    )
    parser.add_argument(
        "--hard-negative-keywords",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\manifests\hard_negative_keywords.txt"),
        help="Text file with one keyword per line for confusing non_UXO samples.",
    )
    parser.add_argument(
        "--hard-negative-factor",
        type=float,
        default=1.0,
        help="Extra sampling multiplier for non_UXO samples matching hard-negative keywords.",
    )
    parser.add_argument(
        "--best-model-out",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\models\underwater_v1_best_model.pth"),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\metrics\underwater_v1_metrics.json"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Data root: {args.data_root}")
    print(f"Aux data root: {args.aux_data_root}")
    print(f"Augmentation profile: {args.augmentation_profile}")
    print(f"Hard negative factor: {args.hard_negative_factor}")

    hard_negative_keywords = load_hard_negative_keywords(args.hard_negative_keywords)
    if hard_negative_keywords:
        print(f"Hard negative keywords: {hard_negative_keywords}")
    else:
        print("Hard negative keywords: []")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    train_tf, eval_tf = build_transforms(augmentation_profile=args.augmentation_profile)
    aux_history = []
    aux_class_counts = {}

    if args.aux_epochs > 0 and args.aux_data_root.exists():
        aux_ds = datasets.ImageFolder(args.aux_data_root, transform=train_tf)
        print(f"Aux classes: {aux_ds.classes}")
        print(f"Aux samples: {len(aux_ds)}")
        aux_class_weights, aux_counts = class_weights_from_trainset(aux_ds)
        aux_class_counts = dict(zip(aux_ds.classes, aux_counts))
        print(f"Aux class counts: {aux_class_counts}")
        print(f"Aux class weights: {aux_class_weights.detach().cpu().tolist()}")
        aux_loader, _ = build_train_loader(
            aux_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            balanced_sampling=args.balanced_sampling,
            class_weights=aux_class_weights,
            hard_negative_keywords=[],
            hard_negative_factor=1.0,
        )
        aux_criterion = nn.CrossEntropyLoss(weight=aux_class_weights.to(device))
        aux_optimizer = optim.Adam(model.parameters(), lr=args.aux_lr)
        aux_history = run_train_epochs(
            model,
            aux_loader,
            aux_criterion,
            aux_optimizer,
            device,
            args.aux_epochs,
            stage_name="Aux",
        )
    elif args.aux_epochs > 0:
        print("Aux stage skipped: aux-data-root does not exist.")

    train_ds = datasets.ImageFolder(args.data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(args.data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(args.data_root / "test", transform=eval_tf)
    print(f"Classes: {train_ds.classes}")
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    class_weights, class_counts = class_weights_from_trainset(train_ds)
    print(f"Train class counts: {dict(zip(train_ds.classes, class_counts))}")
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    train_loader, hard_negative_hits = build_train_loader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampling=args.balanced_sampling,
        class_weights=class_weights,
        hard_negative_keywords=hard_negative_keywords,
        hard_negative_factor=args.hard_negative_factor,
    )
    print(f"Hard negative matches in train set: {hard_negative_hits}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history = []

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
        print(f"Epoch {epoch:02d} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.best_model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.best_model_out)

    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {args.best_model_out}")

    model.load_state_dict(torch.load(args.best_model_out, map_location=device, weights_only=True))
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
    val_cm = confusion_matrix(val_labels, val_preds).tolist()
    test_cm = confusion_matrix(test_labels, test_preds).tolist()

    metrics = {
        "seed": args.seed,
        "augmentation_profile": args.augmentation_profile,
        "aux_data_root": str(args.aux_data_root),
        "aux_epochs": args.aux_epochs,
        "aux_lr": args.aux_lr,
        "aux_class_counts": aux_class_counts,
        "aux_history": aux_history,
        "hard_negative_factor": args.hard_negative_factor,
        "hard_negative_keywords": hard_negative_keywords,
        "best_val_acc": best_val_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "classes": train_ds.classes,
        "train_class_counts": dict(zip(train_ds.classes, class_counts)),
        "history": history,
        "val_confusion_matrix": val_cm,
        "test_confusion_matrix": test_cm,
        "val_report": val_report,
        "test_report": test_report,
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved to: {args.metrics_out}")
    print("\nVal confusion matrix:")
    print(val_cm)
    print("\nTest confusion matrix:")
    print(test_cm)


if __name__ == "__main__":
    main()
