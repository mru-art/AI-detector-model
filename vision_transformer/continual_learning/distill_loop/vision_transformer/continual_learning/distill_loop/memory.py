"""A small exemplar memory: a few image paths per class, stored on disk.

Only paths are stored, never pixels, so the file is tiny. A later update reads the
exemplar images directly and never has to rescan the full old training set.
Selection is random with a fixed seed, so the same images are chosen every time."""
import json
import os
import random


class ExemplarMemory:
    def __init__(self, per_class):
        self.per_class = per_class
        self.items = {}                       # class name -> sorted list of image paths

    def ensure(self, class_name, paths, seed):
        """Pick exemplars for a class, but only if it has none yet. Old classes keep their exemplars."""
        if class_name in self.items or not paths:
            return
        rng = random.Random(f"{seed}:{class_name}")
        self.items[class_name] = sorted(rng.sample(paths, min(self.per_class, len(paths))))

    def labelled(self, class_to_idx):
        """(path, label) pairs for every exemplar, using the global label mapping."""
        return [(p, class_to_idx[c]) for c, paths in self.items.items() for p in paths]

    def missing_files(self):
        return [p for paths in self.items.values() for p in paths if not os.path.isfile(p)]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"per_class": self.per_class, "items": self.items}, f, indent=2)

    @classmethod
    def load(cls, path, per_class):
        with open(path) as f:
            data = json.load(f)
        memory = cls(per_class)
        memory.items = {c: list(paths) for c, paths in data["items"].items()}
        return memory
