#!/usr/bin/env python3
'''Re-split the SAIDO annotation files with a stratified random split.

Why this exists
---------------
classify_scenes.py split the images like this:

    all_imgs = sorted(img_dir.glob('*'))
    i < n_train          -> train
    i < n_train + n_val  -> val
    else                 -> test

That is a split by *file name order*. ImageNet file names are grouped by WNID,
so whole semantic categories land in the same split. The result was a strong
distribution shift between splits: train had scene 1 (animal) at 40 percent
while test had it at 5 percent. That breaks both the scene -> LoRA routing and
the evaluation itself.

This script merges the existing train/val/test annotation files and re-splits
them stratified by (model_label, realfake_label, scene_id) with a fixed seed,
then shuffles each split so the file order is not clustered either. It reuses
the scene_id and caption values already in the files, so CLIP and BLIP are not
re-run.

Usage
-----
    python resplit_annotations.py --dry-run    # report only, write nothing
    python resplit_annotations.py              # back up, then rewrite the files
'''

import argparse
import collections
import os
import random
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ('train', 'val', 'test')
HEADER = 'path\tcaption\tscene_id\t_\tmodel_label\trealfake_label'


def read_records(captions_dir, name):
    path = os.path.join(captions_dir, name + '.txt')
    with open(path, encoding='utf-8') as handle:
        lines = handle.read().splitlines()
    records = []
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) >= 6:
            records.append(parts[:6])
    return records


def allocate(n, train_ratio, val_ratio):
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


def stratified_split(records, seed, train_ratio, val_ratio):
    groups = collections.defaultdict(list)
    for rec in records:
        groups[(rec[4], rec[5], rec[2])].append(rec)
    rng = random.Random(seed)
    out = dict((key, []) for key in SPLITS)
    for key in sorted(groups):
        items = sorted(groups[key])
        rng.shuffle(items)
        n_train, n_val, _ = allocate(len(items), train_ratio, val_ratio)
        out['train'].extend(items[:n_train])
        out['val'].extend(items[n_train:n_train + n_val])
        out['test'].extend(items[n_train + n_val:])
    for key in SPLITS:
        rng.shuffle(out[key])
    return out


def report(split_records, title):
    print('=' * 74)
    print(title)
    print('=' * 74)
    dists = {}
    for name in SPLITS:
        records = split_records[name]
        counter = collections.Counter(int(rec[2]) for rec in records)
        total = sum(counter.values()) or 1
        dists[name] = dict((k, 100.0 * v / total) for k, v in counter.items())
        n_real = sum(1 for rec in records if rec[5] == '0')
        print('%-6s n=%-5d real=%-5d fake=%-5d' % (name, len(records), n_real, len(records) - n_real))
    scenes = sorted(set(k for dist in dists.values() for k in dist))
    print()
    print('%-8s %9s %9s %9s' % ('scene', 'train%', 'val%', 'test%'))
    for scene in scenes:
        print('%-8d %9.1f %9.1f %9.1f' % (
            scene,
            dists['train'].get(scene, 0.0),
            dists['val'].get(scene, 0.0),
            dists['test'].get(scene, 0.0)))
    worst = 0.0
    for scene in scenes:
        worst = max(worst, abs(dists['test'].get(scene, 0.0) - dists['train'].get(scene, 0.0)))
    print()
    print('max |test - train| scene share = %.1f pt' % worst)
    print()
    return worst


def main():
    parser = argparse.ArgumentParser(description='stratified re-split of SAIDO annotations')
    parser.add_argument('--captions-dir', default=os.path.join(HERE, 'dataset', 'captions'))
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=20260101)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    before = dict((name, read_records(args.captions_dir, name)) for name in SPLITS)
    records = [rec for name in SPLITS for rec in before[name]]
    print('loaded %d records from %s' % (len(records), args.captions_dir))
    print()
    gap_before = report(before, 'BEFORE  current files, split by file name order')

    after = stratified_split(records, args.seed, args.train_ratio, args.val_ratio)
    assert sum(len(after[key]) for key in SPLITS) == len(records)
    gap_after = report(after, 'AFTER   stratified by (model, realfake, scene), seed=%d' % args.seed)

    print('scene distribution gap: %.1f pt -> %.1f pt' % (gap_before, gap_after))
    print()

    if args.dry_run:
        print('dry run: nothing written')
        return

    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup_dir = args.captions_dir.rstrip('/\\') + '_backup_' + stamp
    shutil.copytree(args.captions_dir, backup_dir)
    print('backup -> %s' % backup_dir)

    for name in SPLITS:
        path = os.path.join(args.captions_dir, name + '.txt')
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(HEADER + '\n')
            for rec in after[name]:
                handle.write('\t'.join(rec) + '\n')
        print('wrote %s (%d records)' % (path, len(after[name])))


if __name__ == '__main__':
    main()
