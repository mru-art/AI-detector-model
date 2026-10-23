"""NAIVE (fine-tuning): just keep training on whatever experience arrives next.
No memory, no penalty, no teacher. This is the baseline that shows how bad forgetting is."""
import torch.nn.functional as F

from cl_common.trainer import Strategy


class Naive(Strategy):
    def loss(self, model, x, y):
        return F.cross_entropy(model(x), y)
