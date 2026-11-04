"""Bring external AI-vs-human ART datasets into the detector's dataset folder.

The detector trains from  <DATA_DIR>/{train,validation,test}/<class>/  (see dataset.py and
continual_learning/cl_common/config.py). This script fills that layout from art datasets:

  * human-made artworks  ->  the existing  real/  class
  * AI-generated artworks ->  a NEW class per source (e.g. anipro_text2image), so each one can be
                              learned as its own continual-learning experience with add_generator.py

Supported sources
  anipro           AniPrO anime provenance benchmark (Hugging Face: YAN-LIU05/AniPro)
                   real/ -> real, text2image/ -> anipro_text2image, inpainting/ -> anipro_inpainting
  human-ai-artwork Kaggle adamelkholy/human-ai-artwork (270k human + AI artworks, 27 genres)
                   -> real + ai_artwork
  real-fake-art    Kaggle doctorstrange420/real-and-fake-ai-generated-art-images-dataset
                   -> real + ai_art_realfake
  art500k          ART500K (HKUST, human artworks only, request access on their site) -> real
  folder           any other dataset whose folder names (or a CSV) say which images are AI vs human

Typical use (run from the vision_transformer folder)
  1) download        python art_datasets.py download anipro --dest D:/raw/anipro
                     python art_datasets.py download human-ai-artwork --dest D:/raw/human_ai
  2) check labels    python art_datasets.py import human-ai-artwork --src D:/raw/human_ai --dry-run
  3) import          python art_datasets.py import human-ai-artwork --src D:/raw/human_ai --max-per-class 15000
  4) train           add the printed class lines to continual_learning/cl_common/config.py, then
                     python continual_learning/replay/add_generator.py

Nothing in the source folders is changed. Files are hard-linked when possible (no extra disk
space), otherwise copied. Re-running is safe: files already imported are skipped.

Licences: AniPrO is CC BY-NC 4.0 and ART500K is for non-commercial research. Check each Kaggle
dataset's licence page before using a model trained on it commercially.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = (("train", 0.8), ("validation", 0.1), ("test", 0.1))

AI_WORDS = {"ai", "fake", "generated", "aigenerated", "aiart", "synthetic", "gen", "midjourney",
            "dalle", "dalle2", "dalle3", "sd", "sdxl", "stablediffusion", "diffusion", "flux",
            "text2image", "txt2img", "inpainting", "machine"}
HUMAN_WORDS = {"real", "human", "humanart", "humanmade", "original", "authentic", "artist",
               "handmade", "handdrawn", "traditional", "natural"}


# ----------------------------------------------------------------------------- presets
@dataclass
class Source:
    name: str
    ai_class: str | None          # class folder for AI images (None: dataset has no AI images)
    prefix: str                   # file-name prefix, so imported files are easy to find later
    download: str | None          # "hf:<repo>" or "kaggle:<handle>" or None (manual)
    note: str = ""


SOURCES: dict[str, Source] = {
    "anipro": Source("anipro", None, "anipro", "hf:YAN-LIU05/AniPro",
                     "text2image and inpainting become two classes."),
    "human-ai-artwork": Source("human-ai-artwork", "ai_artwork", "haa",
                               "kaggle:adamelkholy/human-ai-artwork"),
    "real-fake-art": Source("real-fake-art", "ai_art_realfake", "rfa",
                            "kaggle:doctorstrange420/real-and-fake-ai-generated-art-images-dataset"),
    "art500k": Source("art500k", None, "art500k", None,
                      "Request access at https://deepart.hkust.edu.hk/ART500K/art500k.html, "
                      "extract it, then import with --src."),
    "folder": Source("folder", "ai_art_other", "ext", None),
}


def default_data_dir() -> str:
    """DATA_DIR from continual_learning/cl_common/config.py, without importing torch."""
    cfg = HERE / "continual_learning" / "cl_common" / "config.py"
    match = re.search(r'^DATA_DIR\s*=\s*["\'](.+?)["\']', cfg.read_text(encoding="utf-8"), re.M)
    return match.group(1) if match else str(HERE / "dataset")


# ----------------------------------------------------------------------------- labelling
def words(part: str) -> set[str]:
    """'AiArtData' -> {ai, art, data, aiartdata}; 'human_made' -> {human, made, humanmade}."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
    tokens = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t]
    return set(tokens) | {"".join(tokens)}


def label_from_path(rel: Path) -> str | None:
    """'ai' / 'human' from the closest folder name that clearly says so, else None."""
    for part in reversed(rel.parent.parts):
        w = words(part)
        ai, human = bool(w & AI_WORDS), bool(w & HUMAN_WORDS)
        if ai != human:
            return "ai" if ai else "human"
    return None


def label_from_value(value: str, ai_value: str | None) -> str | None:
    v = value.strip()
    if ai_value is not None:
        return "ai" if v.lower() == ai_value.lower() else "human"
    if v in {"1", "1.0", "true", "True"}:
        return "ai"
    if v in {"0", "0.0", "false", "False"}:
        return "human"
    w = words(v)
    ai, human = bool(w & AI_WORDS), bool(w & HUMAN_WORDS)
    return None if ai == human else ("ai" if ai else "human")


def iter_images(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "__MACOSX")))
        for f in sorted(filenames):
            if Path(f).suffix.lower() in IMG_EXT and not f.startswith("._"):
                yield Path(dirpath) / f


# Each item: (path, class_name, group). Images in the same group always land in the same split,
# so near-duplicates (e.g. AniPrO's real image and its inpainted copy) never leak into the test set.
def collect_anipro(src: Path, _args) -> list[tuple[Path, str, str]]:
    items = []
    folders = {"real": "real", "text2image": "anipro_text2image", "inpainting": "anipro_inpainting"}
    found = {n: next((p for p in src.rglob(n) if p.is_dir() and p.parent.name != n), None) for n in folders}
    missing = [n for n, p in found.items() if p is None]
    if missing:
        raise SystemExit(f"AniPrO: could not find {missing} folders under {src}. Extract every archive first "
                         "(python art_datasets.py download anipro does this).")
    for name, folder in found.items():
        for p in iter_images(folder):
            rel = p.relative_to(folder)
            sample_id = rel.parts[0] if len(rel.parts) > 1 else rel.stem   # real/12.png | inpainting/12/x.png
            items.append((p, folders[name], f"anipro:{sample_id}"))
    return items


def collect_human_only(src: Path, _args) -> list[tuple[Path, str, str]]:
    return [(p, "real", str(p.relative_to(src))) for p in iter_images(src)]


def collect_labelled(src: Path, args) -> list[tuple[Path, str, str]]:
    ai_class = args.ai_class
    items, unknown = [], []
    if args.csv:
        with open(args.csv, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise SystemExit(f"{args.csv} is empty")
        cols = list(rows[0].keys())
        path_col = args.path_col or next((c for c in cols if c.lower() in {"path", "file", "filename", "image",
                                                                             "image_path", "file_name"}), None)
        label_col = args.label_col or next((c for c in cols if c.lower() in {"label", "class", "target", "is_ai",
                                                                               "ai", "type", "category"}), None)
        if not path_col or not label_col:
            raise SystemExit(f"Pass --path-col and --label-col. CSV columns: {cols}")
        base = Path(args.csv).parent
        for row in rows:
            p = Path(row[path_col])
            p = p if p.is_absolute() else (src / p if (src / p).exists() else base / p)
            label = label_from_value(row[label_col], args.ai_value)
            if label is None or not p.is_file():
                unknown.append(p)
                continue
            items.append((p, ai_class if label == "ai" else "real", str(p)))
    else:
        for p in iter_images(src):
            rel = p.relative_to(src)
            label = label_from_path(rel)
            if label is None:
                unknown.append(rel)
                continue
            items.append((p, ai_class if label == "ai" else "real", str(rel)))
    if unknown:
        print(f"[!] {len(unknown)} images skipped because their label is unclear, e.g.:")
        for u in unknown[:5]:
            print(f"      {u}")
        print("    Use --csv, or rename the folders so they contain words like 'ai'/'fake' or 'human'/'real'.")
    return items


# ----------------------------------------------------------------------------- import
def split_of(group: str, seed: int) -> str:
    h = int(hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for name, share in SPLITS:
        edge += share
        if h < edge:
            return name
    return SPLITS[-1][0]


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different drive / filesystem without hard links -> copy
    shutil.copy2(src, dst)


def run_import(args) -> None:
    source = SOURCES[args.source]
    src = Path(args.src).expanduser()
    if not src.is_dir():
        raise SystemExit(f"--src {src} is not a folder")
    out = Path(args.out or default_data_dir()).expanduser()
    if args.ai_class is None:
        args.ai_class = source.ai_class

    collector = {"anipro": collect_anipro, "art500k": collect_human_only}.get(args.source, collect_labelled)
    items = collector(src, args)
    if not items:
        raise SystemExit("No images found.")

    by_class: dict[str, list] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)
    if args.max_per_class:
        rng = random.Random(args.seed)
        for cls, lst in by_class.items():
            if len(lst) > args.max_per_class:
                by_class[cls] = sorted(rng.sample(lst, args.max_per_class))

    counts: Counter = Counter()
    plan = []
    for cls, lst in sorted(by_class.items()):
        for path, _, group in lst:
            split = split_of(group, args.seed)
            digest = hashlib.sha1(str(path.relative_to(src) if path.is_relative_to(src) else path)
                                  .encode()).hexdigest()[:10]
            name = f"{source.prefix}_{digest}{path.suffix.lower()}"
            plan.append((path, out / split / cls / name, split, cls))
            counts[(split, cls)] += 1

    classes = sorted(by_class)
    print(f"\n{args.source}: {sum(counts.values())} images -> {out}")
    print(f"  {'class':<22}" + "".join(f"{s:>12}" for s, _ in SPLITS))
    for cls in classes:
        print(f"  {cls:<22}" + "".join(f"{counts[(s, cls)]:>12}" for s, _ in SPLITS))
    if args.dry_run:
        print("\nExamples:")
        for cls in classes:
            for path, *_ in by_class[cls][:3]:
                print(f"  {cls:<22} <- {path}")
        print("\nDry run: nothing was written. Remove --dry-run to import.")
        return

    written = skipped = 0
    for i, (path, dst, split, cls) in enumerate(plan, 1):
        if dst.exists() or dst.is_symlink():
            skipped += 1
        else:
            place(path, dst, args.mode)
            written += 1
        if i % 5000 == 0:
            print(f"  ... {i}/{len(plan)}")

    manifest = out / "_imports" / f"{args.source}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "class", "file", "source_path"])
        for path, dst, split, cls in plan:
            w.writerow([split, cls, dst.name, str(path)])
    print(f"\nDone: {written} new, {skipped} already there. Manifest: {manifest}")
    if args.no_hint:
        return
    # Every class folder in the training set, so several imports in a row give one combined hint.
    on_disk = {d.name for d in (out / "train").iterdir() if d.is_dir()} if (out / "train").is_dir() else set()
    print_config_hint(sorted((on_disk | set(classes)) - {"real"}), imported=[c for c in classes if c != "real"])


def print_config_hint(new_classes: list[str], imported: list[str] | None = None) -> None:
    cfg_path = HERE / "continual_learning" / "cl_common" / "config.py"
    text = cfg_path.read_text(encoding="utf-8")
    m = re.search(r"^CLASS_TO_IDX\s*=\s*(\{.*?\})", text, re.M | re.S)
    try:
        current = eval(m.group(1), {}) if m else {}   # plain dict literal in our own config file
    except Exception:
        current = {}
    todo = [c for c in new_classes if c not in current]
    if not todo:
        if imported:
            print("\nThese classes are already in config.py; images were added to them.")
        else:
            print("\nOnly human-made images were added (to real/). Retrain or run add_generator.py with "
                  "the next new generator to use them.")
        return
    nxt = max(current.values(), default=-1) + 1
    new_map = dict(current, **{c: nxt + i for i, c in enumerate(todo)})
    gens = [c for c, _ in sorted(new_map.items(), key=lambda kv: kv[1]) if c != "real"]
    print("\nNext: put these in continual_learning/cl_common/config.py (keep the old order, only append):")
    print(f"  CLASS_TO_IDX = {new_map}")
    print(f"  NUM_CLASSES = {len(new_map)}")
    print(f"  GENERATORS = {gens}")
    print("then teach the saved model only the new classes:")
    print("  python continual_learning/replay/add_generator.py")


# ----------------------------------------------------------------------------- download
def run_download(args) -> None:
    source = SOURCES[args.source]
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    if not source.download:
        raise SystemExit(f"{args.source} can't be downloaded automatically. {source.note}")
    kind, _, handle = source.download.partition(":")
    if kind == "kaggle":
        try:
            import kagglehub
        except ImportError:
            raise SystemExit("pip install kagglehub   (and put your Kaggle API token in ~/.kaggle/kaggle.json)")
        path = Path(kagglehub.dataset_download(handle))
        print(f"Downloaded to {path}")
        if path.resolve() != dest.resolve():
            print(f"Import straight from there:\n  python art_datasets.py import {args.source} --src \"{path}\" --dry-run")
        return
    # Hugging Face: AniPrO ships AES-encrypted zips with one password each (PASSWORDS.csv / .txt).
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub pyzipper")
    raw = dest / "_archives"
    snapshot_download(repo_id=handle, repo_type="dataset", local_dir=raw)
    extract_anipro(raw, dest, delete_archives=args.delete_archives)
    print(f"\nExtracted to {dest}\nNext:  python art_datasets.py import anipro --src \"{dest}\" --dry-run")


def read_passwords(raw: Path) -> dict[str, str]:
    """archive file name -> password, from PASSWORDS.csv or PASSWORDS.txt (any common layout)."""
    pw: dict[str, str] = {}
    files = [f for f in raw.rglob("*") if f.is_file() and f.name.lower().startswith("password")]
    for f in sorted(files):
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            cells = [c.strip() for c in next(csv.reader([line]), []) if c.strip()]
            if len(cells) < 2:   # "name.zip password" / "name.zip: password" / "name.zip = password"
                cells = [c for c in re.split(r"\s*[:=\t]\s*|\s+", line.strip(), maxsplit=1) if c]
            zips = [c for c in cells if c.lower().endswith(".zip")]
            others = [c for c in cells if c not in zips]
            if zips and others:
                pw[Path(zips[0]).name] = others[-1]
    return pw


def extract_anipro(raw: Path, dest: Path, delete_archives: bool = False) -> None:
    passwords = read_passwords(raw)
    archives = sorted(raw.rglob("*.zip"))
    if not archives:
        raise SystemExit(f"No .zip archives under {raw}")
    seven = shutil.which("7z") or shutil.which("7za")
    for archive in archives:
        password = passwords.get(archive.name)
        if password is None:
            print(f"[!] no password listed for {archive.name}; skipped")
            continue
        print(f"Extracting {archive.name} ...")
        if seven:
            import subprocess
            subprocess.run([seven, "x", str(archive), f"-p{password}", f"-o{dest}", "-aos", "-bso0"], check=True)
            if delete_archives:
                archive.unlink()
            continue
        try:
            import pyzipper
        except ImportError:
            raise SystemExit("Install 7-Zip or `pip install pyzipper` to open the encrypted archives.")
        with pyzipper.AESZipFile(archive) as zf:
            zf.setpassword(password.encode())
            zf.extractall(dest)
        if delete_archives:
            archive.unlink()


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download a dataset")
    d.add_argument("source", choices=[k for k, s in SOURCES.items() if s.download])
    d.add_argument("--dest", required=True, help="folder to download/extract into")
    d.add_argument("--delete-archives", action="store_true",
                   help="(anipro) delete each .zip after extracting it, to save disk space")

    i = sub.add_parser("import", help="add a downloaded dataset to the training folder")
    i.add_argument("source", choices=list(SOURCES))
    i.add_argument("--src", required=True, help="the downloaded/extracted dataset folder")
    i.add_argument("--out", help="training dataset folder (default: DATA_DIR in cl_common/config.py)")
    i.add_argument("--ai-class", help="class folder name for this source's AI images")
    i.add_argument("--max-per-class", type=int, default=0, help="cap images per class (0 = all)")
    i.add_argument("--mode", choices=["link", "copy", "symlink"], default="link")
    i.add_argument("--seed", type=int, default=42)
    i.add_argument("--dry-run", action="store_true", help="only show what would be imported")
    i.add_argument("--no-hint", action="store_true", help="don't print the config.py advice (used by the notebook)")
    i.add_argument("--csv", help="(folder/kaggle sources) CSV with image paths and labels")
    i.add_argument("--path-col", help="CSV column with the image path")
    i.add_argument("--label-col", help="CSV column with the label")
    i.add_argument("--ai-value", help="CSV label value that means AI (default: 1/true/'ai'/'fake' ...)")

    sub.add_parser("list", help="show the supported sources")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for k, s in SOURCES.items():
            where = s.download or "manual download"
            print(f"  {k:<18} AI class: {s.ai_class or '-':<18} {where}  {s.note}")
    elif args.cmd == "download":
        run_download(args)
    else:
        run_import(args)


if __name__ == "__main__":
    main()
