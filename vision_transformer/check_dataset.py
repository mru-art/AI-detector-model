"""Sanity-check the dataset folder BEFORE training and warn about shortcuts the model could exploit.
   python check_dataset.py --data_dir D:\\defactify --samples 150"""
import argparse
import os
import random
from collections import Counter

from PIL import Image

from dataset import SPLIT_DIRS, find_split_dir, scan_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--samples", type=int, default=150, help="images inspected per class (train split)")
    a = ap.parse_args()

    for split in SPLIT_DIRS:
        d = find_split_dir(a.data_dir, split)
        names, samples = scan_split(d)
        cnt = Counter(names[i] for _, i in samples)
        print(f"{split:6s} ({os.path.basename(d)}): {len(samples)} images  " + "  ".join(f"{n}={cnt[n]}" for n in names))

    names, samples = scan_split(find_split_dir(a.data_dir, "train"))
    rng = random.Random(0)
    print("\nImage properties (sampled from train):")
    summary = {}
    for ci, name in enumerate(names):
        files = [p for p, i in samples if i == ci]
        rng.shuffle(files)
        sizes, fmts = [], Counter()
        for p in files[:a.samples]:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
                    fmts[im.format] += 1
            except Exception as e:
                print(f"  unreadable: {p} ({e})")
        top = Counter(sizes).most_common(2)
        print(f"  {name:11s} formats={dict(fmts)}  common sizes={top}")
        summary[name] = (set(fmts), {s for s, _ in Counter(sizes).most_common(5)})

    real = next((n for n in names if n.lower() == "real"), None)
    if real:
        fake_fmts = set().union(*[summary[n][0] for n in names if n != real])
        fake_sizes = set().union(*[summary[n][1] for n in names if n != real])
        if not (summary[real][0] & fake_fmts):
            print("\n[!] File format alone separates real from AI images -> a model can cheat on it. "
                  "The default 'crop' preprocessing + random JPEG augmentation reduces this.")
        if not (summary[real][1] & fake_sizes):
            print("[!] Image resolution alone separates real from AI images. The default 'crop' mode keeps "
                  "native pixels (resolution shortcut stays partly available); 'resize' hides it but "
                  "also hides generator artefacts. Keep it in mind when judging test accuracy.")
    print("\nDataset layout OK.")


if __name__ == "__main__":
    main()
