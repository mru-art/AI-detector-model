"""EWC (Elastic Weight Consolidation): some weights matter a lot for what the model already knows.
After each experience we measure how important every weight is (the "Fisher"), and while learning
the next experience we punish moving the important weights away from where they were:

        loss = cross_entropy + EWC_LAMBDA / 2 * sum( importance * (weight - old_weight)^2 )

Important weights are held in place, unimportant weights are free to learn the new generator."""
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cl_common import config as cfg
from cl_common.trainer import Strategy


class EWC(Strategy):
    def __init__(self):
        self.importance = None      # name -> tensor: how important each weight is (summed over experiences)
        self.old_weights = None     # name -> tensor: the weights at the end of the previous experience
        self.penalty = 0.0

    def loss(self, model, x, y):
        loss = F.cross_entropy(model(x), y)
        if self.importance is None:             # first experience: no penalty yet
            return loss
        penalty = 0.0
        for name, p in model.named_parameters():
            penalty = penalty + (self.importance[name] * (p - self.old_weights[name]) ** 2).sum()
        penalty = cfg.EWC_LAMBDA / 2 * penalty
        self.penalty = penalty.item()           # only for the log: compare it with the cross-entropy part
        return loss + penalty

    def after_experience(self, model, dataset):
        device = torch.device(cfg.DEVICE)
        model.eval()                            # measure importance without dropout noise
        names = [n for n, _ in model.named_parameters()]
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        # Importance of a weight = average of (its gradient)^2 over some images, one image at a time.
        picks = random.Random(cfg.SEED).sample(range(len(dataset)), min(cfg.EWC_FISHER_SAMPLES, len(dataset)))
        loader = DataLoader(Subset(dataset, picks), batch_size=1, num_workers=cfg.NUM_WORKERS // 2)
        for x, y in loader:
            model.zero_grad()
            F.cross_entropy(model(x.to(device)), y.to(device)).backward()
            for n, p in model.named_parameters():
                fisher[n] += p.grad.detach() ** 2 / len(picks)
        model.zero_grad(set_to_none=True)

        # Rescale so the average importance is 1 (raw values are tiny), then add to what we had before.
        mean = sum(f.sum() for f in fisher.values()) / sum(f.numel() for f in fisher.values())
        for n in names:
            fisher[n] /= mean
            if self.importance is not None:
                fisher[n] += self.importance[n]
        self.importance = fisher
        self.old_weights = {n: p.detach().clone() for n, p in model.named_parameters()}

    def status(self):
        return f"(EWC penalty part: {self.penalty:.4f})" if self.importance is not None else ""
