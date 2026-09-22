#!/usr/bin/env python3
"""CLIP scene classification + BLIP caption generation for SAIDO annotation files

Usage:
  # CLIP scene classification only (caption = scene class name)
  python classify_scenes.py

  # CLIP scene classification + BLIP descriptive caption
  python classify_scenes.py --use_caption

  # Specify custom BLIP model path
  python classify_scenes.py --use_caption --blip_model_path ./models/BLIP
"""
import os, argparse
import sys
import collections
import random
import torch
from pathlib import Path
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import warnings
warnings.filterwarnings("ignore")

from transformers import BlipProcessor, BlipForConditionalGeneration

PROJECT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = PROJECT_DIR / "dataset"
CAPTIONS_DIR = DATASET_DIR / "captions"

SCENE_CLASSES = [
    "an activity or action",
    "a type of animal",
    "a building or structure",
    "a piece of clothing",
    "a type of food or dish",
    "a natural scene or landscape",
    "a common object",
    "a person or human figure",
    "a kind of plant",
    "a type of vehicle",
]

MODEL_MAP = {
    "ADM": 0, "GLIDE": 1, "SAGAN": 2, "ProGAN": 3,
    "VQDM": 4, "BigGAN": 5, "wukong": 6,
    "diffusion1.5": 7, "midjourney": 8,
}


def classify_images_clip(image_dir, clip_model, clip_processor, device):
    scene_texts = [f"a photo of {s}" for s in SCENE_CLASSES]
    text_inputs = clip_processor(text=scene_texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    results = {}
    image_paths = sorted(image_dir.glob("*")) if image_dir.exists() else []
    for img_path in image_paths:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            image_input = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)
            scene_id = similarity.argmax().item()
            results[img_path.name] = scene_id
        except Exception as e:
            print(f"skip {img_path.name}: {e}")
            results[img_path.name] = 0
    return results


def generate_blip_caption(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)


def write_annotation_file(txt_path, records, model_map,
                          blip_processor=None, blip_model=None, device="cpu"):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    use_caption = blip_processor is not None and blip_model is not None
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("path\tcaption\tscene_id\t_\tmodel_label\trealfake_label\n")
        for img_path, scene_id, model_label, rf_label in tqdm(records, desc=f"writing {txt_path.name}"):
            if use_caption:
                caption = generate_blip_caption(str(img_path), blip_processor, blip_model, device)
            else:
                caption = SCENE_CLASSES[scene_id]
            f.write(f"{img_path.as_posix()}\t{caption}\t{scene_id}\t_\t{model_label}\t{rf_label}\n")


def allocate_split_counts(n, train_ratio, val_ratio):
    '''Split n items of one stratum into (train, val, test) counts.'''
    if n <= 2:
        return n, 0, 0
    if n < 7:
        return n - 2, 1, 1
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_val < 1:
        n_val = 1
    if n_train + n_val > n - 1:
        n_train = n - n_val - 1
    if n_train < 1:
        n_train = 1
        n_val = max(0, n - 2)
    return n_train, n_val, n - n_train - n_val


def stratified_split_records(records, seed=20260101,
                             train_ratio=0.70, val_ratio=0.15):
    '''Split by (model_label, realfake_label, scene_id).

    A plain file-name-order split does NOT keep train/val/test comparable:
    ImageNet file names are grouped by WNID, so whole semantic categories end
    up in a single split, and the scene distribution of test drifts far away
    from train.'''
    groups = collections.defaultdict(list)
    for rec in records:
        groups[(str(rec[2]), str(rec[3]), str(rec[1]))].append(rec)
    rng = random.Random(seed)
    train_set, val_set, test_set = [], [], []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda r: r[0].as_posix())
        rng.shuffle(items)
        n_train, n_val, _ = allocate_split_counts(len(items), train_ratio, val_ratio)
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)
    return train_set, val_set, test_set


def main():
    print("=" * 60)
    print("SAIDO Scene Classification + Annotation (v3 -- BLIP caption)")
    print("=" * 60)
    print(f"Local CLIP model: {str(PROJECT_DIR / 'CLIP14')}")

    parser = argparse.ArgumentParser(description="SAIDO scene classification + annotation")
    parser.add_argument("--use_caption", action="store_true",
                        help="Use BLIP to generate descriptive captions")
    parser.add_argument("--blip_model_path", type=str, default=None,
                        help="Path to BLIP model folder")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (auto: cuda/cpu)")
    parser.add_argument("--seed", type=int, default=20260101,
                        help="Seed of the stratified train/val/test split")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Train ratio of the stratified split")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation ratio of the stratified split")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print("\n[1/3] Loading CLIP model...")
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(str(PROJECT_DIR / "CLIP14"), local_files_only=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(str(PROJECT_DIR / "CLIP14"))
    clip_model.eval()
    print("  CLIP model loaded")

    blip_processor = None
    blip_model = None

    if args.use_caption:
        blip_path = args.blip_model_path or str(PROJECT_DIR / "models" / "BLIP")
        print(f"\n[1.5/3] Loading BLIP model...")
        print(f"  Model path: {blip_path}")
        if not os.path.exists(blip_path):
            print(f"  ERROR: BLIP model not found: {blip_path}")
            sys.exit(1)
        blip_processor = BlipProcessor.from_pretrained(blip_path, local_files_only=True)
        blip_model = BlipForConditionalGeneration.from_pretrained(blip_path, local_files_only=True)
        blip_model.to(device)
        blip_model.eval()
        print("  BLIP model loaded")
    else:
        print("\n  [NOTE] Without --use_caption, captions use scene class names")

    print("\n[2/3] CLIP scene classification...")
    for split_name in ["train", "val", "test"]:
        old_txt = CAPTIONS_DIR / f"{split_name}.txt"
        if old_txt.exists():
            os.remove(old_txt)
            print(f"  removed old {split_name}.txt")

    for model_name in MODEL_MAP:
        for label_name in ["real", "fake"]:
            img_dir = DATASET_DIR / label_name / model_name
            if not img_dir.exists():
                continue
            rf_label = 0 if label_name == "real" else 1
            n_imgs = len(list(img_dir.glob("*")))
            scene_map = classify_images_clip(img_dir, clip_model, clip_processor, device)
            scene_file = img_dir / ".scene_ids.txt"
            with open(scene_file, "w") as f:
                for img_name, sid in scene_map.items():
                    f.write(f"{img_name}\t{sid}\n")

    print("\n[3/3] Generating annotation files...")
    all_records = []

    for model_name, model_id in MODEL_MAP.items():
        for label_name, rf_label in [("real", 0), ("fake", 1)]:
            img_dir = DATASET_DIR / label_name / model_name
            scene_file = img_dir / ".scene_ids.txt"
            if not scene_file.exists():
                continue
            scene_map = {}
            with open(scene_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        scene_map[parts[0]] = int(parts[1])
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                if img_path.name not in scene_map:
                    continue
                all_records.append((img_path, scene_map[img_path.name], model_id, rf_label))

    train_records, val_records, test_records = stratified_split_records(
        all_records, seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    print("  stratified split: train=%d val=%d test=%d" % (
        len(train_records), len(val_records), len(test_records)))

    for model_name in MODEL_MAP:
        for label_name in ["real", "fake"]:
            sf = DATASET_DIR / label_name / model_name / ".scene_ids.txt"
            if sf.exists():
                sf.unlink()

    if args.use_caption:
        print("  Generating BLIP captions...")
        write_annotation_file(CAPTIONS_DIR / "train.txt", train_records, MODEL_MAP,
                              blip_processor, blip_model, device)
        write_annotation_file(CAPTIONS_DIR / "val.txt", val_records, MODEL_MAP,
                              blip_processor, blip_model, device)
        write_annotation_file(CAPTIONS_DIR / "test.txt", test_records, MODEL_MAP,
                              blip_processor, blip_model, device)
    else:
        write_annotation_file(CAPTIONS_DIR / "train.txt", train_records, MODEL_MAP)
        write_annotation_file(CAPTIONS_DIR / "val.txt", val_records, MODEL_MAP)
        write_annotation_file(CAPTIONS_DIR / "test.txt", test_records, MODEL_MAP)

    print(f"\n  Done!")
    print(f"  Caption mode: {'BLIP caption' if args.use_caption else 'scene class name'}")
    print(f"  train.txt: {len(train_records)} records")
    print(f"  val.txt:   {len(val_records)} records")
    print(f"  test.txt:  {len(test_records)} records")
    print(f"\n  Next: python training.py --yaml mytrain.yaml")

if __name__ == "__main__":
    main()
