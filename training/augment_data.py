"""
Apple Dataset Augmentation Script (with Bounding Box Support)
=============================================================
Augments BOTH images AND their YOLO-format bounding box labels together.

Your dataset structure (from the zip):
    labels/train/IMG_XXXX.txt    ← YOLO labels (class x_center y_center w h)
    data.yaml                    ← Class definition (0: apple_cluster)

Your images:
    Immagini/IMG_XXXX.jpeg       ← Original photos

This script:
    1. Reads each image + its matching label file
    2. Applies augmentations to BOTH image and bounding boxes simultaneously
    3. Outputs augmented images + labels in YOLO-compatible directory structure

Usage:
    python training/augment_data.py

Output structure (ready for YOLO training):
    dataset/
    ├── images/
    │   ├── train/     ← originals + augmented (~80%)
    │   └── val/       ← originals + augmented (~20%)
    ├── labels/
    │   ├── train/     ← matching label files
    │   └── val/       ← matching label files
    └── data.yaml      ← YOLO config file
"""

import os
import cv2
import random
import shutil
import numpy as np
from pathlib import Path

try:
    import albumentations as A
except ImportError:
    print("=" * 60)
    print("ERROR: albumentations is not installed.")
    print("Run:  uv add albumentations opencv-python-headless")
    print("=" * 60)
    raise


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent

# Source directories
IMAGES_DIR = str(project_root / "Immagini")
LABELS_DIR = str(project_root / "Database_apples" / "Apples_dataset" / "labels" / "train")

# Output directory (YOLO-ready structure)
OUTPUT_DIR = str(project_root / "dataset")

# How many augmented copies per original image
AUGMENTATIONS_PER_IMAGE = 8

# Train/Val split ratio
VAL_SPLIT = 0.2  # 20% of originals go to validation

# Random seed for reproducibility
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────
# YOLO label I/O helpers
# ──────────────────────────────────────────────────────────────
def read_yolo_labels(label_path: str) -> tuple[list[list[float]], list[int]]:
    """
    Reads a YOLO-format label file.

    Returns:
        bboxes: list of [x_center, y_center, width, height] (all 0-1 normalized)
        class_ids: list of integer class IDs
    """
    bboxes = []
    class_ids = []

    if not os.path.exists(label_path):
        return bboxes, class_ids

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            bboxes.append([x_center, y_center, w, h])
            class_ids.append(class_id)

    return bboxes, class_ids


def write_yolo_labels(label_path: str, bboxes: list, class_ids: list):
    """Writes bounding boxes and class IDs to a YOLO-format label file."""
    with open(label_path, "w") as f:
        for bbox, cls_id in zip(bboxes, class_ids):
            x_center, y_center, w, h = bbox
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


# ──────────────────────────────────────────────────────────────
# Augmentation Pipelines (with bbox support)
# ──────────────────────────────────────────────────────────────
# Every pipeline is wrapped with A.BboxParams so that bounding
# boxes are transformed in lockstep with the image.

BBOX_PARAMS = A.BboxParams(
    format="yolo",                # YOLO format: [x_center, y_center, width, height]
    label_fields=["class_labels"],
    min_visibility=0.3,           # Drop boxes that become <30% visible after crop/rotate
    min_area=100,                 # Drop boxes smaller than 100px² (too small to detect)
)


def get_augmentation_pipelines():
    """
    Returns augmentation pipelines tailored for orchard apple images.
    Each pipeline transforms BOTH the image and bounding boxes.
    """

    pipelines = []

    # ── 1. LIGHTING VARIATIONS ──────────────────────────────
    # Simulates different times of day, cloud cover, shadows
    pipelines.append(("lighting", A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.3, 0.3),
                p=1.0
            ),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
        ], p=1.0),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.5
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            src_radius=100,
            p=0.3
        ),
    ], bbox_params=BBOX_PARAMS)))

    # ── 2. GEOMETRIC TRANSFORMS ─────────────────────────────
    # Different camera angles, perspectives
    pipelines.append(("geometric", A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=(-25, 25), border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            shear=(-10, 10),
            mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
    ], bbox_params=BBOX_PARAMS)))

    # ── 3. COLOR / HUE SHIFTS ───────────────────────────────
    # Critical: green apples vs green leaves
    pipelines.append(("color", A.Compose([
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0
            ),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05, p=1.0),
        ], p=1.0),
    ], bbox_params=BBOX_PARAMS)))

    # ── 4. BLUR / FOCUS SIMULATION ──────────────────────────
    # Autofocus variation
    pipelines.append(("blur_focus", A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.5), p=1.0),
        ], p=0.8),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=0.3),
    ], bbox_params=BBOX_PARAMS)))

    # ── 5. WEATHER SIMULATION ───────────────────────────────
    # Fog and rain
    pipelines.append(("weather", A.Compose([
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.1, p=1.0),
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=15, drop_width=1,
                drop_color=(180, 180, 180),
                blur_value=3, brightness_coefficient=0.8,
                rain_type="drizzle", p=1.0
            ),
        ], p=0.7),
        A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15), p=0.5),
    ], bbox_params=BBOX_PARAMS)))

    # ── 6. NOISE / QUALITY DEGRADATION ──────────────────────
    # Low-light noise, compression artifacts
    pipelines.append(("noise", A.Compose([
        A.OneOf([
            A.GaussNoise(std_range=(10.0 / 255.0, 40.0 / 255.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=1.0),
        ], p=0.8),
        A.ImageCompression(quality_range=(40, 85), p=0.4),
        A.Downscale(scale_range=(0.5, 0.75), p=0.3),
    ], bbox_params=BBOX_PARAMS)))

    # ── 7. CROP & ZOOM ──────────────────────────────────────
    # Closer/farther from the tree
    pipelines.append(("crop_zoom", A.Compose([
        A.RandomResizedCrop(
            size=(640, 640),
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
    ], bbox_params=BBOX_PARAMS)))

    # ── 8. HEAVY / COMBINED ─────────────────────────────────
    # Multiple mild augmentations stacked
    pipelines.append(("heavy_combined", A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(std_range=(5.0 / 255.0, 20.0 / 255.0), p=1.0),
        ], p=0.3),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            p=0.3
        ),
    ], bbox_params=BBOX_PARAMS)))

    return pipelines


# ──────────────────────────────────────────────────────────────
# Main augmentation loop
# ──────────────────────────────────────────────────────────────
def augment_dataset():
    """
    Reads all images + YOLO labels, applies augmentations to BOTH,
    and saves everything in a YOLO-ready directory structure with
    train/val split.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Create output directory structure ──
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # ── Collect all image files that have labels ──
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    all_images = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    # Match images to labels
    paired = []
    unpaired = []
    for img_file in all_images:
        basename = os.path.splitext(img_file)[0]
        label_file = basename + ".txt"
        label_path = os.path.join(LABELS_DIR, label_file)
        if os.path.exists(label_path):
            paired.append((img_file, label_file))
        else:
            unpaired.append(img_file)

    print(f"📸 Found {len(all_images)} total images")
    print(f"🏷️  Found {len(paired)} images with labels")
    if unpaired:
        print(f"⚠️  {len(unpaired)} images without labels (will be skipped): "
              f"{unpaired[:5]}{'...' if len(unpaired) > 5 else ''}")
    print(f"🔄 Creating {AUGMENTATIONS_PER_IMAGE} augmentations per image")
    print(f"📊 Train/Val split: {(1 - VAL_SPLIT) * 100:.0f}% / {VAL_SPLIT * 100:.0f}%")
    print("=" * 60)

    # ── Shuffle and split into train/val ──
    random.shuffle(paired)
    val_count = max(1, int(len(paired) * VAL_SPLIT))
    val_pairs = paired[:val_count]
    train_pairs = paired[val_count:]

    print(f"📂 Train: {len(train_pairs)} originals → "
          f"~{len(train_pairs) * (1 + AUGMENTATIONS_PER_IMAGE)} total")
    print(f"📂 Val:   {len(val_pairs)} originals → "
          f"~{len(val_pairs) * (1 + AUGMENTATIONS_PER_IMAGE)} total")
    print("=" * 60)

    # Load pipelines
    pipelines = get_augmentation_pipelines()

    # Resize transform (preserves bbox coordinates since they're normalized)
    resize = A.Compose([
        A.LongestMaxSize(max_size=640),
        A.PadIfNeeded(
            min_height=640,
            min_width=640,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
    ], bbox_params=BBOX_PARAMS)

    total_created = 0
    failed = 0

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        print(f"\n{'─' * 30} {split_name.upper()} {'─' * 30}")

        img_out_dir = os.path.join(OUTPUT_DIR, "images", split_name)
        lbl_out_dir = os.path.join(OUTPUT_DIR, "labels", split_name)

        for idx, (img_file, lbl_file) in enumerate(split_pairs, 1):
            basename = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]

            # Read image
            img_path = os.path.join(IMAGES_DIR, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ⚠️  Could not read {img_file}, skipping.")
                failed += 1
                continue

            # Read labels
            lbl_path = os.path.join(LABELS_DIR, lbl_file)
            bboxes, class_ids = read_yolo_labels(lbl_path)

            # ── Save resized original ──
            try:
                resized = resize(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_ids
                )
                resized_img = resized["image"]
                resized_bboxes = resized["bboxes"]
                resized_classes = resized["class_labels"]

                # Save original (resized)
                cv2.imwrite(
                    os.path.join(img_out_dir, f"{basename}_original{ext}"),
                    resized_img
                )
                write_yolo_labels(
                    os.path.join(lbl_out_dir, f"{basename}_original.txt"),
                    resized_bboxes,
                    resized_classes
                )
                total_created += 1
            except Exception as e:
                print(f"  ⚠️  Error resizing {img_file}: {e}")
                failed += 1
                continue

            # ── Apply augmentations ──
            for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
                pipe_name, pipe_transform = random.choice(pipelines)

                try:
                    augmented = pipe_transform(
                        image=resized_img,
                        bboxes=resized_bboxes,
                        class_labels=resized_classes
                    )

                    aug_image = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_classes = augmented["class_labels"]

                    # Skip if all bounding boxes were lost (e.g., cropped out)
                    if len(aug_bboxes) == 0:
                        continue

                    # Ensure correct output size
                    if aug_image.shape[0] != 640 or aug_image.shape[1] != 640:
                        aug_image = cv2.resize(aug_image, (640, 640))

                    # Save augmented image + label
                    out_name = f"{basename}_aug{aug_idx:02d}_{pipe_name}"
                    cv2.imwrite(
                        os.path.join(img_out_dir, f"{out_name}{ext}"),
                        aug_image
                    )
                    write_yolo_labels(
                        os.path.join(lbl_out_dir, f"{out_name}.txt"),
                        aug_bboxes,
                        aug_classes
                    )
                    total_created += 1

                except Exception as e:
                    print(f"  ⚠️  Error augmenting {img_file} "
                          f"(aug {aug_idx}, '{pipe_name}'): {e}")
                    failed += 1

            # Progress
            if idx % 10 == 0 or idx == len(split_pairs):
                print(f"  ✅ {split_name}: {idx}/{len(split_pairs)} images processed")

    # ── Write data.yaml for YOLO training ──
    data_yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: apple_cluster\n")

    # ── Summary ──
    train_imgs = len(os.listdir(os.path.join(OUTPUT_DIR, "images", "train")))
    val_imgs = len(os.listdir(os.path.join(OUTPUT_DIR, "images", "val")))

    print("\n" + "=" * 60)
    print("✅ Augmentation complete!")
    print(f"   Train images: {train_imgs}")
    print(f"   Val images:   {val_imgs}")
    print(f"   Total:        {train_imgs + val_imgs}")
    if failed:
        print(f"   ⚠️  Failures: {failed}")
    print(f"\n   Output: {OUTPUT_DIR}")
    print(f"   Config: {data_yaml_path}")
    print("=" * 60)
    print("\n🚀 Ready for YOLO training! Next step:")
    print("   uv run yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100")


if __name__ == "__main__":
    augment_dataset()
