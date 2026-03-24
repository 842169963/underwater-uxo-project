import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")

POSITIVE_DIRS = {
    ("Examples_munition_objects_TOI", "corroded_shell"),
    ("Examples_munition_objects_TOI", "examples%20bombs"),
    ("Examples_munition_objects_TOI", "torpedos"),
}

NEGATIVE_DIRS = {
    ("Examples_other_objects_NONTOI", "GeoEcoMar_NONTOI", "animals"),
    ("Examples_other_objects_NONTOI", "GeoEcoMar_NONTOI", "equipment"),
    ("Examples_other_objects_NONTOI", "GeoEcoMar_NONTOI", "mussels"),
    ("Examples_other_objects_NONTOI", "GeoEcoMar_NONTOI", "natural%20rocks"),
    ("Examples_other_objects_NONTOI", "GeoEcoMar_NONTOI", "sediment"),
    ("Examples_other_objects_NONTOI", "GeoEcoMar_NONTOI", "wooden%20branch"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "boxes"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "calibation%20frame"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "Iron%20barrel"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "iron%20cabinet"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "iron%20cylinder"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "natural%20rocks"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "plank"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "wooden%20branch"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "wooden%20ship%20wreck"),
    ("Examples_other_objects_NONTOI", "Images_NONTOI", "wreck"),
}

GROUP_PATTERNS = [
    r"(toi_box\d+)",
    r"(shipwreck\d+)",
    r"(torpedo\d+)",
    r"(coraltorpedo\d+)",
    r"(bomb\d+)",
    r"(shell\d+)",
    r"(wreck\d+)",
    r"(box\d+)",
    r"(rock\d+)",
    r"(cali\d+)",
    r"(plank\d+)",
    r"(branch\d+)",
    r"(cylinder\d+)",
    r"(barrel\d+)",
    r"(cabinet\d+)",
    r"(mussel\d+)",
    r"(animal\d+)",
    r"(object\d+)",
    r"(obj\d+)",
]


def path_label(rel_parts):
    rel_tuple = tuple(rel_parts)
    if rel_tuple[:2] in POSITIVE_DIRS:
        return "UXO"
    if rel_tuple[:3] in NEGATIVE_DIRS:
        return "non_UXO"
    return None


def is_optical_candidate(path):
    if "non optical" in [p.lower() for p in path.parts]:
        return False
    if "bv" in path.name.lower():
        return False
    return True


def focus_score(image_arr):
    gray = (
        0.299 * image_arr[..., 0]
        + 0.587 * image_arr[..., 1]
        + 0.114 * image_arr[..., 2]
    ).astype(np.float32)
    pad = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    lap = (
        pad[:-2, 1:-1]
        + pad[1:-1, :-2]
        - 4.0 * pad[1:-1, 1:-1]
        + pad[1:-1, 2:]
        + pad[2:, 1:-1]
    )
    return float(lap.var())


def extract_group_id(file_stem, parent_name):
    text = file_stem.lower().replace("%20", "_").replace(" ", "_")
    for pat in GROUP_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    prefix = re.sub(r"[^a-z0-9_]+", "", text)[:24]
    if prefix:
        return f"{parent_name.lower()}_{prefix}"
    return f"{parent_name.lower()}_unknown"


def resize_aspect_with_padding(img, target=224, pad_color=(0, 0, 0)):
    """Isotropic resize + centered padding to avoid geometric distortion."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return img.resize((target, target), Image.BILINEAR)

    scale = float(target) / float(max(w, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), pad_color)
    x = (target - nw) // 2
    y = (target - nh) // 2
    canvas.paste(img, (x, y))
    return canvas


def assign_groups_balanced(group_items, ratios):
    split_counts = {s: 0 for s in SPLITS}
    split_groups = {s: [] for s in SPLITS}
    total = sum(cnt for _, cnt in group_items)
    target = {s: total * ratios[s] for s in SPLITS}

    for gid, cnt in sorted(group_items, key=lambda x: x[1], reverse=True):
        # Put next group where current fill ratio is smallest.
        best_split = min(
            SPLITS,
            key=lambda s: (
                split_counts[s] / target[s] if target[s] > 0 else 1e9,
                split_counts[s],
            ),
        )
        split_groups[best_split].append(gid)
        split_counts[best_split] += cnt

    return split_groups, split_counts, target


def collect_items(data_root):
    items = []
    for p in data_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if not is_optical_candidate(p):
            continue
        rel = p.relative_to(data_root)
        label = path_label(rel.parts)
        if label is None:
            continue

        try:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
                arr = np.asarray(rgb)
                score = focus_score(arr)
                w, h = rgb.size
        except Exception:
            continue

        group_id = extract_group_id(p.stem, rel.parts[-2])
        items.append(
            {
                "path": p,
                "rel_path": str(rel),
                "label": label,
                "group_id": group_id,
                "width": w,
                "height": h,
                "focus_score": score,
            }
        )
    return items


def build_manifest(items, seed):
    rng = random.Random(seed)
    rng.shuffle(items)

    scores = np.array([it["focus_score"] for it in items], dtype=np.float64)
    p5 = float(np.percentile(scores, 5))
    p10 = float(np.percentile(scores, 10))

    for it in items:
        if it["focus_score"] <= p5:
            qflag = "low_focus_hard"
        elif it["focus_score"] <= p10:
            qflag = "low_focus_soft"
        else:
            qflag = "normal"
        it["quality_flag"] = qflag

    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    label_groups = defaultdict(lambda: defaultdict(list))
    for idx, it in enumerate(items):
        key = f'{it["label"]}:{it["group_id"]}'
        label_groups[it["label"]][key].append(idx)

    split_for_index = {}
    split_report = {}

    for label in ("UXO", "non_UXO"):
        group_items = [(gid, len(idxs)) for gid, idxs in label_groups[label].items()]
        groups, counts, target = assign_groups_balanced(group_items, ratios)
        split_report[label] = {"counts": counts, "target": target, "groups": groups}
        for split, gids in groups.items():
            for gid in gids:
                for idx in label_groups[label][gid]:
                    split_for_index[idx] = split

    for idx, it in enumerate(items):
        it["split"] = split_for_index[idx]

    return items, split_report, {"p5": p5, "p10": p10}


def write_manifest(items, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "rel_path",
        "label",
        "split",
        "group_id",
        "focus_score",
        "quality_flag",
        "width",
        "height",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in sorted(items, key=lambda x: (x["split"], x["label"], x["rel_path"])):
            w.writerow(
                {
                    "file": str(it["path"]),
                    "rel_path": it["rel_path"],
                    "label": it["label"],
                    "split": it["split"],
                    "group_id": it["group_id"],
                    "focus_score": f'{it["focus_score"]:.4f}',
                    "quality_flag": it["quality_flag"],
                    "width": it["width"],
                    "height": it["height"],
                }
            )


def build_dataset(items, out_root):
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in SPLITS:
        for label in ("UXO", "non_UXO"):
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    for i, it in enumerate(items):
        src = it["path"]
        dst_name = f'{it["group_id"]}__{Path(it["rel_path"]).name}'
        dst = out_root / it["split"] / it["label"] / dst_name
        with Image.open(src) as im:
            img = im.convert("RGB")
            img = resize_aspect_with_padding(img, target=224)
            img.save(dst, quality=95)

        # Keep a heartbeat for long runs.
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(items)} images ...")


def summarize(items):
    counts = defaultdict(int)
    qcounts = defaultdict(int)
    for it in items:
        counts[(it["split"], it["label"])] += 1
        qcounts[(it["split"], it["quality_flag"])] += 1

    print("\n=== Split/Class Summary ===")
    for split in SPLITS:
        print(
            f"{split}: UXO={counts[(split, 'UXO')]} | "
            f"non_UXO={counts[(split, 'non_UXO')]}"
        )

    print("\n=== Split/Quality Summary ===")
    for split in SPLITS:
        print(
            f"{split}: normal={qcounts[(split, 'normal')]} | "
            f"low_focus_soft={qcounts[(split, 'low_focus_soft')]} | "
            f"low_focus_hard={qcounts[(split, 'low_focus_hard')]}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare underwater v1 dataset with group split and hard-case flags."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\tuc_images"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\manifests\underwater_dataset_v1_manifest.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\underwater_dataset_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only generate manifest and summary; do not materialize image folders.",
    )
    args = parser.parse_args()

    print(f"Scanning images in: {args.data_root}")
    items = collect_items(args.data_root)
    if not items:
        print("No eligible images found.")
        return

    print(f"Collected eligible optical images: {len(items)}")
    items, split_report, quantiles = build_manifest(items, seed=args.seed)
    write_manifest(items, args.manifest)
    print(f"Manifest saved to: {args.manifest}")
    print(
        f"Quality thresholds: p5={quantiles['p5']:.4f}, "
        f"p10={quantiles['p10']:.4f}"
    )

    if not args.skip_build:
        print(f"Building dataset into: {args.output_root}")
        build_dataset(items, args.output_root)
        print("Dataset build completed.")

    summarize(items)

    print("\n=== Group Split Targets (image count) ===")
    for label in ("UXO", "non_UXO"):
        target = split_report[label]["target"]
        counts = split_report[label]["counts"]
        print(
            f"{label}: train {counts['train']:.0f}/{target['train']:.1f}, "
            f"val {counts['val']:.0f}/{target['val']:.1f}, "
            f"test {counts['test']:.0f}/{target['test']:.1f}"
        )


if __name__ == "__main__":
    main()
