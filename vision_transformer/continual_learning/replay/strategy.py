"""REPLAY: keep a small memory of images from old experiences and mix them into every new experience.

 - after each experience we keep a few images of the generator that was just learned
   (the total memory is fixed: REPLAY_MEMORY_SIZE, shared equally between the generators seen so far)
 - while training on a new experience, the old images are repeated until there are as many of them as
   new images, so every batch is about half new data, half old data
 ("real" needs no memory: it is already part of every experience)"""
import random

import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from cl_common import config as cfg
from cl_common.data import ImageDataset, make_loader
from cl_common.trainer import Strategy


class Replay(Strategy):
    def __init__(self):
        self.memory = {}                       # generator label -> list of (image_path, label)
        self.rng = random.Random(cfg.SEED)

    def make_loader(self, dataset):
        if not self.memory:                    # first experience: nothing to replay yet
            return super().make_loader(dataset)
        old = ImageDataset([item for items in self.memory.values() for item in items], train=True)
        copies = max(1, len(dataset) // len(old))          # repeat memory so it is about as big as the new data
        return make_loader(ConcatDataset([dataset] + [old] * copies), train=True)

    def loss(self, model, x, y):
        return F.cross_entropy(model(x), y)    # same loss as naive - the only change is WHICH images we train on

    def after_experience(self, model, dataset):
        new = [item for item in dataset.items if item[1] != cfg.CLASS_TO_IDX["real"]]
        self.memory[new[0][1]] = new                                   # add the generator we just learned ...
        per_generator = cfg.REPLAY_MEMORY_SIZE // len(self.memory)
        for label, items in self.memory.items():                       # ... and shrink everyone to an equal share
            self.memory[label] = self.rng.sample(items, min(per_generator, len(items)))
        print(f"  replay memory: {sum(len(v) for v in self.memory.values())} images "
              f"({per_generator} per generator)")
