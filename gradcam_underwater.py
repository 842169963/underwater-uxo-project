import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import datasets, models, transforms


def resize_aspect_with_padding(img, target=224, pad_color=(0, 0, 0)):
    w, h = img.size
    if w <= 0 or h <= 0:
        return img.resize((target, target), Image.BILINEAR)
    scale = float(target) / float(max(w, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), pad_color)
    canvas.paste(img, ((target - nw) // 2, (target - nh) // 2))
    return canvas


def preprocess_for_model(img):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tf(img).unsqueeze(0)


def heat_colormap(v):
    # v in [0, 1]
    r = np.clip(2.0 * v - 0.5, 0, 1)
    g = np.clip(2.0 * (1.0 - np.abs(v - 0.5)), 0, 1)
    b = np.clip(1.5 - 2.0 * v, 0, 1)
    return np.stack([r, g, b], axis=-1)


def build_gradcam(model, input_tensor, target_class, target_layer):
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    score = logits[:, target_class].sum()
    score.backward()

    fh.remove()
    bh.remove()

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam, logits.detach()


def blend_overlay(base_img, cam, alpha=0.45):
    base = np.asarray(base_img).astype(np.float32) / 255.0
    color = heat_colormap(cam).astype(np.float32)
    out = (1.0 - alpha) * base + alpha * color
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\models\underwater_v1_best_model_seed42.pth"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\outputs\gradcam"),
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only-misclassified",
        action="store_true",
        help="Only export samples where prediction != ground truth.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV path to save per-sample summary.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = datasets.ImageFolder(args.data_root / args.split, transform=None)
    class_names = ds.classes
    sample_paths = [Path(p) for p, _ in ds.samples]
    sample_labels = [y for _, y in ds.samples]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.layer4[-1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(sample_paths)))
    random.shuffle(indices)

    print(f"Using split={args.split}, total={len(sample_paths)}")
    print(f"Output dir: {args.out_dir}")

    exported = 0
    rows = []
    for idx in indices:
        if exported >= args.num_samples:
            break

        path = sample_paths[idx]
        true_id = sample_labels[idx]
        true_name = class_names[true_id]

        with Image.open(path) as im:
            base = resize_aspect_with_padding(im.convert("RGB"), target=224)

        x = preprocess_for_model(base).to(device)
        with torch.enable_grad():
            cam, logits = build_gradcam(model, x, target_class=true_id, target_layer=target_layer)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_id = int(np.argmax(probs))
        pred_name = class_names[pred_id]
        pred_prob = float(probs[pred_id])
        is_error = pred_id != true_id

        rows.append(
            {
                "file": str(path),
                "split": args.split,
                "true_label": true_name,
                "pred_label": pred_name,
                "pred_prob": pred_prob,
                "is_error": int(is_error),
            }
        )

        if args.only_misclassified and not is_error:
            continue

        overlay = blend_overlay(base, cam, alpha=0.45)
        canvas = Image.new("RGB", (224 * 2, 224 + 28), (10, 10, 10))
        canvas.paste(base, (0, 0))
        canvas.paste(overlay, (224, 0))

        draw = ImageDraw.Draw(canvas)
        txt = (
            f"true={true_name} | pred={pred_name} ({pred_prob:.3f}) | "
            f"file={path.name}"
        )
        draw.text((6, 228), txt, fill=(235, 235, 235))

        out_name = (
            f"{exported+1:02d}_{args.split}_{true_name}_pred-{pred_name}_"
            f"{path.stem[:40]}.png"
        )
        canvas.save(args.out_dir / out_name)
        exported += 1

    if args.summary_csv is not None:
        import csv

        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "split",
                    "true_label",
                    "pred_label",
                    "pred_prob",
                    "is_error",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    n_error = sum(r["is_error"] for r in rows)
    print(
        f"Grad-CAM generation completed. Exported={exported}, "
        f"evaluated={len(rows)}, errors={n_error}."
    )


if __name__ == "__main__":
    main()
