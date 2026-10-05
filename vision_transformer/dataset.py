import io
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True  # don't crash on a half-written file

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SPLIT_DIRS = {"train": ["train"], "val": ["validation", "val", "valid"], "test": ["test"]}


# --------------------------------------------------------------------------
# transforms (all picklable classes -> safe with Windows multiprocessing)
# --------------------------------------------------------------------------
class PadToMin:
    """Pad (black) images smaller than `size` so that a crop is always possible."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if w >= self.size and h >= self.size:
            return img
        nw, nh = max(w, self.size), max(h, self.size)
        canvas = Image.new("RGB", (nw, nh))
        canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
        return canvas


class RandomJPEG:
    def __init__(self, p, qmin=60, qmax=100):
        self.p, self.qmin, self.qmax = p, qmin, qmax

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(self.qmin, self.qmax))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    def __init__(self, p, smin=0.1, smax=1.5):
        self.p, self.smin, self.smax = p, smin, smax

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(self.smin, self.smax)))


class FiveCropStack:
    """4 corners + centre, each normalised -> tensor [5, 3, H, W] (test-time augmentation)."""

    def __init__(self, size):
        self.crop = T.FiveCrop(size)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __call__(self, img):
        return torch.stack([self.norm(c) for c in self.crop(img)])


def build_transform(preprocess, size, train, tta=False, jpeg_prob=0.0, blur_prob=0.0):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if preprocess == "crop":
        if train:
            # NOTE: no rotation / affine / colour-jitter on purpose: interpolation and colour
            # changes destroy the pixel-level traces the detector is supposed to learn.
            return T.Compose([PadToMin(size), T.RandomCrop(size), T.RandomHorizontalFlip(),
                              RandomBlur(blur_prob), RandomJPEG(jpeg_prob)] + norm)
        if tta:
            return T.Compose([PadToMin(size), FiveCropStack(size)])
        return T.Compose([PadToMin(size), T.CenterCrop(size)] + norm)
    # resize mode
    resize = T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    if train:
        return T.Compose([resize, T.RandomHorizontalFlip(), RandomBlur(blur_prob),
                          RandomJPEG(jpeg_prob)] + norm)
    return T.Compose([resize] + norm)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def find_split_dir(root, split):
    for name in SPLIT_DIRS[split]:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"Could not find a '{SPLIT_DIRS[split][0]}' folder inside '{root}'. "
        f"Expected layout: {root}/train|validation|test/<real|dalle3|midjourney|sd21|sd3|sdxl>/")


def scan_split(split_dir):
    """-> sorted source-class names, list of (path, source_idx)"""
    names = sorted(d.name for d in os.scandir(split_dir) if d.is_dir())
    if not names:
        raise FileNotFoundError(f"No class sub-folders in {split_dir}")
    samples = []
    for idx, name in enumerate(names):
        with os.scandir(os.path.join(split_dir, name)) as it:
            files = sorted(e.path for e in it
                           if e.is_file() and os.path.splitext(e.name)[1].lower() in IMG_EXT)
        samples += [(f, idx) for f in files]
    return names, samples


class DefactifyDataset(Dataset):
    """Returns (image_tensor, label, source_idx).
    source_idx always indexes the 6 generator folders (used for per-generator reporting);
    label is the training target (== source_idx for multiclass, 0/1 for binary)."""

    def __init__(self, root, split, task, transform, max_per_class=0, seed=42):
        self.split, self.task, self.transform = split, task, transform
        self.source_names, samples = scan_split(find_split_dir(root, split))
        self.real_idx = next((i for i, n in enumerate(self.source_names) if n.lower() == "real"), None)
        if task == "binary" and self.real_idx is None:
            raise ValueError("Binary task needs a 'real' folder")

        if max_per_class > 0:
            rng = random.Random(seed)
            kept = []
            for c in range(len(self.source_names)):
                cls = [s for s in samples if s[1] == c]
                rng.shuffle(cls)
                kept += cls[:max_per_class]
            samples = sorted(kept)

        self.paths = [p for p, _ in samples]
        self.source_targets = np.array([s for _, s in samples], dtype=np.int64)
        if task == "binary":
            self.targets = (self.source_targets != self.real_idx).astype(np.int64)
            self.class_names = ["real", "ai_generated"]
        else:
            self.targets = self.source_targets.copy()
            self.class_names = list(self.source_names)

    def __len__(self):
        return len(self.paths)

    def _open(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return im.convert("RGB")
        except Exception:
            return None

    def __getitem__(self, i):
        img = self._open(i)
        tries = 0
        while img is None and tries < 5:  # corrupt file: substitute a random sample instead of crashing
            print(f"[warn] unreadable image skipped: {self.paths[i]}")
            i = random.randrange(len(self.paths))
            img = self._open(i)
            tries += 1
        if img is None:
            raise RuntimeError("Too many unreadable images")
        return self.transform(img), int(self.targets[i]), int(self.source_targets[i])

    def class_counts(self):
        return np.bincount(self.targets, minlength=len(self.class_names))


def make_loader(cfg, split, train, tta=False):
    transform = build_transform(cfg.preprocess, cfg.image_size, train, tta=tta,
                                jpeg_prob=cfg.jpeg_prob, blur_prob=cfg.blur_prob)
    ds = DefactifyDataset(cfg.data_dir, split, cfg.task, transform, cfg.max_per_class, cfg.seed)
    nw = cfg.num_workers
    if not train and nw > 1:
        nw = max(1, nw // 2)  # evaluation loaders get half the workers: train workers stay alive, so total RAM matters
    # TTA batches are 5x bigger -> use a smaller batch to keep memory the same
    bs = max(1, cfg.batch_size // 5) if tta else cfg.batch_size
    kwargs = dict(batch_size=bs, shuffle=train, drop_last=train, num_workers=nw,
                  pin_memory=torch.cuda.is_available())
    if nw > 0:
        kwargs.update(persistent_workers=train, prefetch_factor=2)
    return ds, DataLoader(ds, **kwargs)
