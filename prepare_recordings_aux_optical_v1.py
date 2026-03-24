import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


TOPLEVEL_LABELS = {
    "100lbs_bomb": "UXO",
    "100lbs_bomb_(floor)": "UXO",
    "15cm_mortar": "UXO",
    "20lbs_incendiary": "UXO",
    "test_cylinder": "non_UXO",
}

JSON_CLASS_TO_SUBCLASS = {
    "100lbs_aircraft_bomb": "100lbs_aircraft_bomb",
    "mortar_shell": "mortar_shell",
    "incindiary_grenade": "incindiary_grenade",
}

# Labels were generated for the SD frames. The bundled JPGs are FHD, so coords
# must be multiplied by 3 to land on the right pixels in the optical images.
DEFAULT_BBOX_SCALE = 3.0

# Cylinder runs have no labels. This weak box is centered on the object region
# observed across the labelled runs, but expanded to keep useful background.
DEFAULT_CYLINDER_BOX = (0.24, 0.33, 0.76, 0.98)


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    uxo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a small optical auxiliary dataset from recordings/."
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=project_root / "recordings",
        help="Root recordings directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=uxo_root / "aux_recordings_optical_v1",
        help="Where cropped auxiliary images will be written.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=uxo_root / "manifests" / "recordings_aux_optical_v1_manifest.csv",
        help="CSV manifest describing all exported crops.",
    )
    parser.add_argument(
        "--samples-per-run",
        type=int,
        default=5,
        help="How many frames to keep from each labelled UXO run.",
    )
    parser.add_argument(
        "--cylinder-samples-per-run",
        type=int,
        default=10,
        help="How many frames to keep from each cylinder run.",
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=DEFAULT_BBOX_SCALE,
        help="Scale factor applied to json bbox coordinates.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.18,
        help="Extra context margin around labelled boxes.",
    )
    parser.add_argument(
        "--cylinder-box",
        type=str,
        default="0.24,0.33,0.76,0.98",
        help="Weak crop box for cylinder runs as x1,y1,x2,y2 in [0,1].",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for exported crops.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove the previous exported auxiliary dataset before writing a new one.",
    )
    return parser.parse_args()


def parse_notes(notes_path):
    info = {"trajectory": "", "tilt": "", "roll": ""}
    if not notes_path.exists():
        return info
    text = notes_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- Trajectory:"):
            info["trajectory"] = line.replace("- Trajectory:", "").strip()
        elif line.startswith("- Tilt:"):
            info["tilt"] = line.replace("- Tilt:", "").strip()
        elif line.startswith("- roll:"):
            info["roll"] = line.replace("- roll:", "").strip()
    return info


def choose_evenly_spaced(items, k):
    if not items or k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[len(items) // 2]]
    idxs = []
    max_idx = len(items) - 1
    for i in range(k):
        idx = round(i * max_idx / (k - 1))
        idxs.append(idx)
    dedup = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            dedup.append(items[idx])
            seen.add(idx)
    return dedup


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_box(box, img_w, img_h, margin):
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    mx = w * margin
    my = h * margin
    x1 = clamp(int(round(x1 - mx)), 0, img_w - 1)
    y1 = clamp(int(round(y1 - my)), 0, img_h - 1)
    x2 = clamp(int(round(x2 + mx)), x1 + 1, img_w)
    y2 = clamp(int(round(y2 + my)), y1 + 1, img_h)
    return x1, y1, x2, y2


def weak_box_from_norm(norm_box, img_w, img_h):
    x1n, y1n, x2n, y2n = norm_box
    x1 = clamp(int(round(x1n * img_w)), 0, img_w - 1)
    y1 = clamp(int(round(y1n * img_h)), 0, img_h - 1)
    x2 = clamp(int(round(x2n * img_w)), x1 + 1, img_w)
    y2 = clamp(int(round(y2n * img_h)), y1 + 1, img_h)
    return x1, y1, x2, y2


def load_json_bbox(json_path, bbox_scale):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    box = (
        float(data["x_min"]) * bbox_scale,
        float(data["y_min"]) * bbox_scale,
        float(data["x_max"]) * bbox_scale,
        float(data["y_max"]) * bbox_scale,
    )
    return data["class"], box


def save_crop(src_img, dst_img, crop_box, jpeg_quality):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_img) as im:
        rgb = im.convert("RGB")
        crop = rgb.crop(crop_box)
        crop.save(dst_img, quality=jpeg_quality)


def iter_runs(recordings_root):
    for target_dir in sorted(recordings_root.iterdir()):
        if not target_dir.is_dir():
            continue
        if target_dir.name not in TOPLEVEL_LABELS:
            continue
        for run_dir in sorted(target_dir.iterdir()):
            if run_dir.is_dir():
                yield target_dir.name, run_dir


def collect_positive_samples(run_dir):
    gopro_dir = run_dir / "gopro"
    labels_dir = run_dir / "labels"
    samples = []
    if not gopro_dir.exists() or not labels_dir.exists():
        return samples
    for json_path in sorted(labels_dir.glob("*.json")):
        img_path = gopro_dir / f"{json_path.stem}.jpg"
        if img_path.exists():
            samples.append((img_path, json_path))
    return samples


def collect_cylinder_samples(run_dir):
    gopro_dir = run_dir / "gopro"
    if not gopro_dir.exists():
        return []
    return [(img_path, None) for img_path in sorted(gopro_dir.glob("*.jpg"))]


def main():
    args = parse_args()
    recordings_root = args.recordings_root.resolve()
    output_root = args.output_root.resolve()
    manifest_out = args.manifest_out.resolve()
    cylinder_box = tuple(float(x) for x in args.cylinder_box.split(","))
    if len(cylinder_box) != 4:
        raise ValueError("--cylinder-box must have 4 comma-separated floats")

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    stats = {"runs": 0, "exported": 0}

    for target_name, run_dir in iter_runs(recordings_root):
        label = TOPLEVEL_LABELS[target_name]
        notes = parse_notes(run_dir / "notes.txt")
        stats["runs"] += 1

        if target_name == "test_cylinder":
            candidates = collect_cylinder_samples(run_dir)
            selected = choose_evenly_spaced(candidates, args.cylinder_samples_per_run)
        else:
            candidates = collect_positive_samples(run_dir)
            selected = choose_evenly_spaced(candidates, args.samples_per_run)

        for img_path, json_path in selected:
            group_id = f"{target_name}/{run_dir.name}"
            frame_id = img_path.stem
            rel_name = f"{target_name}__{run_dir.name}__{frame_id}.jpg"

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            if json_path is not None:
                subclass, raw_box = load_json_bbox(json_path, args.bbox_scale)
                subclass = JSON_CLASS_TO_SUBCLASS.get(subclass, subclass)
                crop_box = expand_box(raw_box, img_w, img_h, args.crop_margin)
                crop_type = "bbox_x3_margin"
            else:
                subclass = "cylinder"
                crop_box = weak_box_from_norm(cylinder_box, img_w, img_h)
                crop_type = "weak_box"

            dst_path = output_root / label / subclass / rel_name
            save_crop(img_path, dst_path, crop_box, args.jpeg_quality)

            rows.append(
                {
                    "label": label,
                    "subclass": subclass,
                    "target_name": target_name,
                    "run_id": run_dir.name,
                    "group_id": group_id,
                    "frame_id": frame_id,
                    "trajectory": notes["trajectory"],
                    "tilt": notes["tilt"],
                    "roll": notes["roll"],
                    "source_image": str(img_path),
                    "json_label": str(json_path) if json_path else "",
                    "crop_type": crop_type,
                    "x_min": crop_box[0],
                    "y_min": crop_box[1],
                    "x_max": crop_box[2],
                    "y_max": crop_box[3],
                    "file": str(dst_path),
                }
            )
            stats["exported"] += 1

    fieldnames = [
        "label",
        "subclass",
        "target_name",
        "run_id",
        "group_id",
        "frame_id",
        "trajectory",
        "tilt",
        "roll",
        "source_image",
        "json_label",
        "crop_type",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "file",
    ]

    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {stats['exported']} crops from {stats['runs']} runs.")
    print(f"Output root: {output_root}")
    print(f"Manifest: {manifest_out}")


if __name__ == "__main__":
    main()
