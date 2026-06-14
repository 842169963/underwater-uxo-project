import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


TRASH_CLASS_MAP = {
    0: "plastic",
    1: "bio",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_yolo_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls_id = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:])
    return cls_id, cx, cy, w, h


def normalized_box_to_pixels(cx, cy, w, h, width, height):
    bw = w * width
    bh = h * height
    x1 = int(round((cx * width) - bw / 2.0))
    y1 = int(round((cy * height) - bh / 2.0))
    x2 = int(round((cx * width) + bw / 2.0))
    y2 = int(round((cy * height) + bh / 2.0))
    return x1, y1, x2, y2


def expand_box(box, width, height, margin):
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    mx = int(round(bw * margin))
    my = int(round(bh * margin))
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(width, x2 + mx)
    ny2 = min(height, y2 + my)
    if nx2 <= nx1:
        nx2 = min(width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(height, ny1 + 1)
    return nx1, ny1, nx2, ny2


def evenly_spaced_indices(length, wanted):
    if length <= 0 or wanted <= 0:
        return []
    if length <= wanted:
        return list(range(length))
    if wanted == 1:
        return [length // 2]

    raw = []
    for i in range(wanted):
        idx = round(i * (length - 1) / float(wanted - 1))
        raw.append(int(idx))

    seen = set()
    result = []
    for idx in raw:
        if idx not in seen:
            result.append(idx)
            seen.add(idx)

    cursor = 0
    while len(result) < wanted and cursor < length:
        if cursor not in seen:
            result.append(cursor)
            seen.add(cursor)
        cursor += 1

    return sorted(result[:wanted])


def copy_recordings_aux(source_root: Path, output_root: Path, manifest_rows):
    copied = 0
    for image_path in source_root.rglob("*.jpg"):
        rel = image_path.relative_to(source_root)
        destination = output_root / rel
        ensure_dir(destination.parent)
        shutil.copy2(image_path, destination)

        top_parts = rel.parts
        label = top_parts[0]
        subclass = top_parts[1] if len(top_parts) > 1 else ""
        manifest_rows.append(
            {
                "source": "recordings_aux",
                "label": label,
                "subclass": subclass,
                "group_id": image_path.stem.split("__")[1] if "__" in image_path.stem else image_path.stem,
                "file": str(destination),
                "origin_file": str(image_path),
            }
        )
        copied += 1
    return copied


def collect_trash_candidates(trash_root: Path):
    candidates = defaultdict(list)

    for txt_path in trash_root.rglob("*.txt"):
        image_path = txt_path.with_suffix(".jpg")
        if not image_path.exists():
            continue

        stem = txt_path.stem
        if "_frame" not in stem:
            continue
        sequence_id, frame_id = stem.split("_frame", 1)

        for obj_idx, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines()):
            parsed = parse_yolo_line(line)
            if parsed is None:
                continue
            cls_id, cx, cy, w, h = parsed
            if cls_id not in TRASH_CLASS_MAP:
                continue
            candidates[(TRASH_CLASS_MAP[cls_id], sequence_id)].append(
                {
                    "image_path": image_path,
                    "txt_path": txt_path,
                    "class_name": TRASH_CLASS_MAP[cls_id],
                    "sequence_id": sequence_id,
                    "frame_id": int(frame_id),
                    "obj_idx": obj_idx,
                    "bbox_norm": (cx, cy, w, h),
                }
            )

    for key in candidates:
        candidates[key].sort(key=lambda item: (item["frame_id"], item["obj_idx"]))

    return candidates


def export_trash_negatives(candidates, output_root: Path, manifest_rows, margin, per_sequence_limits):
    exported = Counter()

    max_seq_by_class = per_sequence_limits.get("_max_sequences", {})
    allowed_sequences = {}
    grouped_sequences = defaultdict(list)
    for (class_name, sequence_id) in candidates.keys():
        grouped_sequences[class_name].append(sequence_id)

    for class_name, sequence_ids in grouped_sequences.items():
        unique_ids = sorted(set(sequence_ids))
        max_seq = max_seq_by_class.get(class_name)
        if max_seq is None or max_seq <= 0 or max_seq >= len(unique_ids):
            allowed_sequences[class_name] = set(unique_ids)
            continue
        keep_idx = evenly_spaced_indices(len(unique_ids), max_seq)
        allowed_sequences[class_name] = {unique_ids[i] for i in keep_idx}

    for (class_name, sequence_id), items in sorted(candidates.items()):
        if sequence_id not in allowed_sequences.get(class_name, set()):
            continue
        wanted = per_sequence_limits.get(class_name, 0)
        if wanted <= 0:
            continue

        chosen_indices = evenly_spaced_indices(len(items), wanted)
        for idx in chosen_indices:
            item = items[idx]
            with Image.open(item["image_path"]) as im:
                rgb = im.convert("RGB")
                width, height = rgb.size
                x1, y1, x2, y2 = normalized_box_to_pixels(*item["bbox_norm"], width, height)
                x1, y1, x2, y2 = expand_box((x1, y1, x2, y2), width, height, margin)
                crop = rgb.crop((x1, y1, x2, y2))

            out_dir = output_root / "non_UXO" / class_name
            ensure_dir(out_dir)
            out_name = f"{sequence_id}__frame{item['frame_id']:07d}__obj{item['obj_idx']:02d}.jpg"
            out_path = out_dir / out_name
            crop.save(out_path, quality=95)

            manifest_rows.append(
                {
                    "source": "trash_icra19",
                    "label": "non_UXO",
                    "subclass": class_name,
                    "group_id": f"trash_{class_name}_{sequence_id}",
                    "file": str(out_path),
                    "origin_file": str(item["image_path"]),
                }
            )
            exported[class_name] += 1

    return exported


def main():
    parser = argparse.ArgumentParser(description="Build a combined auxiliary optical dataset from recordings + trash hard negatives.")
    parser.add_argument(
        "--recordings-aux-root",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\aux_recordings_optical_v1"),
    )
    parser.add_argument(
        "--trash-root",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\trash_ICRA19\dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\aux_combined_optical_v1"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path(r"C:\Users\stephenxxy\Desktop\project\uxo_project\manifests\aux_combined_optical_v1_manifest.csv"),
    )
    parser.add_argument("--margin", type=float, default=0.18)
    parser.add_argument("--plastic-per-sequence", type=int, default=1)
    parser.add_argument("--bio-per-sequence", type=int, default=2)
    parser.add_argument("--plastic-max-sequences", type=int, default=0)
    parser.add_argument("--bio-max-sequences", type=int, default=0)
    parser.add_argument("--clear-output", action="store_true")
    args = parser.parse_args()

    if args.clear_output:
        clear_dir(args.output_root)
    else:
        ensure_dir(args.output_root)

    manifest_rows = []

    copied = copy_recordings_aux(args.recordings_aux_root, args.output_root, manifest_rows)
    trash_candidates = collect_trash_candidates(args.trash_root)
    exported = export_trash_negatives(
        trash_candidates,
        args.output_root,
        manifest_rows,
        margin=args.margin,
        per_sequence_limits={
            "plastic": args.plastic_per_sequence,
            "bio": args.bio_per_sequence,
            "_max_sequences": {
                "plastic": args.plastic_max_sequences,
                "bio": args.bio_max_sequences,
            },
        },
    )

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "label", "subclass", "group_id", "file", "origin_file"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_counts = Counter()
    subclass_counts = Counter()
    for row in manifest_rows:
        total_counts[row["label"]] += 1
        subclass_counts[row["subclass"]] += 1

    print(f"Copied recordings aux images: {copied}")
    print(f"Exported trash negatives: {dict(exported)}")
    print(f"Total label counts: {dict(total_counts)}")
    print(f"Subclass counts: {dict(subclass_counts)}")
    print(f"Combined dataset: {args.output_root}")
    print(f"Manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
