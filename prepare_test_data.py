#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAIDO 小型测试数据集创建脚本 v2
===============================
从桌面 archive / archive1 取图片，
按 SAIDO 要求的结构整理到 ./dataset/ 目录，
并正确划分 train/val/test。
"""

import os, random, shutil
from pathlib import Path

# ========== 配置 ==========
# 每个生成器每种标签取多少张 (real/fake 各取 N 张)
TRAIN_N = 140    # 训练
VAL_N   = 30    # 验证
TEST_N  = 30    # 测试
# 共 TRAIN_N+VAL_N+TEST_N = 100 对 per generator

USERPROFILE = os.environ.get("USERPROFILE", r"C:\Users\86186")
SRC_ARCHIVE  = Path(USERPROFILE) / "Desktop/archive"
SRC_ARCHIVE1 = Path(USERPROFILE) / "Desktop/archive1/HybridForensics_Dataset_High"
DST          = Path(__file__).parent.resolve() / "dataset"

# ========== SAIDO 生成器映射 ==========
MODEL_MAP = {
    "ADM": 0, "GLIDE": 1, "SAGAN": 2, "ProGAN": 3,
    "VQDM": 4, "BigGAN": 5, "wukong": 6,
    "diffusion1.5": 7, "midjourney": 8,
}

def collect_images(src_dirs, n):
    """
    从多个 src_dir 中收集图片，随机取 n 张返回列表
    src_dirs: list of Path
    """
    all_files = []
    for d in src_dirs:
        if d and d.is_dir():
            for f in d.rglob("*"):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    all_files.append(f)
    if not all_files:
        return []
    return random.sample(all_files, min(n, len(all_files)))


def copy_to(images, dst_dir):
    """将图片列表复制到目标目录"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in images:
        shutil.copy2(f, dst_dir / f.name)
    return len(images)


def write_annotation(txt_path, records):
    """
    写入标注文件
    records: list of (image_path, scene_id, model_label, realfake_label)
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("path\tcaption\tscene_id\t_\tmodel_label\trealfake_label\n")
        for img_path, scene_id, model_label, rf_label in records:
            caption = "test"
            f.write(f"{img_path.as_posix()}\t{caption}\t{scene_id}\t_\t{model_label}\t{rf_label}\n")


# ========== 主流程 ==========

def main():
    random.seed(42)
    print("=" * 60)
    print("SAIDO 小型测试数据集创建 v2")
    print("=" * 60)

    # ---- 图片复制阶段 ----
    print("\n[1/2] 从各生成器取图并复制到 dataset/...")

    # GenImage 7 个生成器: real 来自 train/nature + val/nature, fake 来自 train/ai + val/ai
    genimage_folders = {
        "BigGAN":       ("imagenet_ai_0419_biggan", "ai", "nature"),
        "VQDM":         ("imagenet_ai_0419_vqdm", "ai", "nature"),
        "diffusion1.5": ("imagenet_ai_0424_sdv5", "ai", "nature"),
        "wukong":       ("imagenet_ai_0424_wukong", "ai", "nature"),
        "ADM":          ("imagenet_ai_0508_adm", "ai", "nature"),
        "GLIDE":        ("imagenet_glide", "ai", "nature"),
        "midjourney":   ("imagenet_midjourney", "ai", "nature"),
    }

    for name, (folder, fake_sub, real_sub) in genimage_folders.items():
        base = SRC_ARCHIVE / folder

        # fake: train/ai + val/ai → 分 train/val/test
        fake_train = collect_images([base / "train" / fake_sub], TRAIN_N)
        fake_val   = collect_images([base / "val"   / fake_sub], VAL_N)
        fake_test  = collect_images([base / "val"   / fake_sub], TEST_N)

        # real: train/nature + val/nature
        real_train = collect_images([base / "train" / real_sub], TRAIN_N)
        real_val   = collect_images([base / "val"   / real_sub], VAL_N)
        real_test  = collect_images([base / "val"   / real_sub], TEST_N)

        # 复制到统一目录 (方便后面生成标注文件)
        copy_to(fake_train + fake_val + fake_test, DST / "fake" / name)
        copy_to(real_train + real_val + real_test, DST / "real" / name)

        print(f"  [OK] {name}: real={TRAIN_N+VAL_N+TEST_N}, fake={TRAIN_N+VAL_N+TEST_N}")

    # ---- archive1: ProGAN, StyleGAN3->SAGAN ----
    # ProGAN
    fake_train = collect_images([SRC_ARCHIVE1 / "Fake_GAN/ProGAN"], TRAIN_N)
    fake_val   = collect_images([SRC_ARCHIVE1 / "Fake_GAN/ProGAN"], VAL_N)
    fake_test  = collect_images([SRC_ARCHIVE1 / "Fake_GAN/ProGAN"], TEST_N)
    real_train = collect_images([SRC_ARCHIVE1 / "Real/MS_COCO"], TRAIN_N)
    real_val   = collect_images([SRC_ARCHIVE1 / "Real/MS_COCO"], VAL_N)
    real_test  = collect_images([SRC_ARCHIVE1 / "Real/MS_COCO"], TEST_N)
    copy_to(fake_train + fake_val + fake_test, DST / "fake/ProGAN")
    copy_to(real_train + real_val + real_test, DST / "real/ProGAN")
    print(f"  [OK] ProGAN: real={TRAIN_N+VAL_N+TEST_N}, fake={TRAIN_N+VAL_N+TEST_N}")

    # StyleGAN3 -> SAGAN (人脸用 FFHQ 做 real)
    fake_train = collect_images([SRC_ARCHIVE1 / "Fake_GAN/StyleGAN3"], TRAIN_N)
    fake_val   = collect_images([SRC_ARCHIVE1 / "Fake_GAN/StyleGAN3"], VAL_N)
    fake_test  = collect_images([SRC_ARCHIVE1 / "Fake_GAN/StyleGAN3"], TEST_N)
    real_train = collect_images([SRC_ARCHIVE1 / "Real/FFHQ"], TRAIN_N)
    real_val   = collect_images([SRC_ARCHIVE1 / "Real/FFHQ"], VAL_N)
    real_test  = collect_images([SRC_ARCHIVE1 / "Real/FFHQ"], TEST_N)
    copy_to(fake_train + fake_val + fake_test, DST / "fake/SAGAN")
    copy_to(real_train + real_val + real_test, DST / "real/SAGAN")
    print(f"  [OK] SAGAN: real={TRAIN_N+VAL_N+TEST_N}, fake={TRAIN_N+VAL_N+TEST_N}")

    # ---- 生成标注阶段 ----
    print("\n[2/2] 生成 train.txt / val.txt / test.txt...")

    scene_id = 0  # 小测试统一用场景 0

    def make_records(model_name, model_id, real_imgs, fake_imgs):
        records = []
        for img in real_imgs:
            records.append((img, scene_id, model_id, 0))  # 0 = real
        for img in fake_imgs:
            records.append((img, scene_id, model_id, 1))  # 1 = fake
        return records

    train_records = []
    val_records = []
    test_records = []

    for name, mid in MODEL_MAP.items():
        base_real = DST / "real" / name
        base_fake = DST / "fake" / name

        all_real = sorted(base_real.glob("*")) if base_real.exists() else []
        all_fake = sorted(base_fake.glob("*")) if base_fake.exists() else []

        train_records += make_records(name, mid, all_real[:TRAIN_N], all_fake[:TRAIN_N])
        val_records   += make_records(name, mid, all_real[TRAIN_N:TRAIN_N+VAL_N], all_fake[TRAIN_N:TRAIN_N+VAL_N])
        test_records  += make_records(name, mid, all_real[TRAIN_N+VAL_N:], all_fake[TRAIN_N+VAL_N:])

    write_annotation(DST / "captions/train.txt", train_records)
    write_annotation(DST / "captions/val.txt", val_records)
    write_annotation(DST / "captions/test.txt", test_records)

    print(f"  train.txt: {len(train_records)} 条 (每生成器 real={TRAIN_N}, fake={TRAIN_N})")
    print(f"  val.txt:   {len(val_records)} 条 (每生成器 real={VAL_N}, fake={VAL_N})")
    print(f"  test.txt:  {len(test_records)} 条 (每生成器 real={TEST_N}, fake={TEST_N})")

    # ---- 最终统计 ----
    print("\n" + "=" * 60)
    print("最终数据集结构:")
    print("=" * 60)
    for name in MODEL_MAP:
        n_real = len(list((DST / "real" / name).glob("*"))) if (DST / "real" / name).exists() else 0
        n_fake = len(list((DST / "fake" / name).glob("*"))) if (DST / "fake" / name).exists() else 0
        print(f"  {name:15s}: real={n_real:3d}, fake={n_fake:3d}")

    total = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    print(f"\n总大小: {total / 1024 / 1024:.1f} MB")
    print(f"\n下一步运行训练:")
    print(f"  cd {Path(__file__).parent.resolve()}")
    print("  python training.py --yaml mytrain.yaml")

if __name__ == "__main__":
    main()
