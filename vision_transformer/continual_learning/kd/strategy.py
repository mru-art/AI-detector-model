"""KD (Knowledge Distillation, a.k.a. LwF): before learning a new generator, freeze a copy of the model
as a TEACHER. While the student learns the new images, it must also keep giving the same answers
as the teacher about the OLD classes:

        loss = cross_entropy(new labels) + KD_LAMBDA * KL(teacher's old-class answer || student's old-class answer)

Only the old classes' scores are compared, so this never fights against learning the new generator."""
import copy

import torch
import torch.nn.functional as F

from cl_common import config as cfg
from cl_common.trainer import Strategy


class KD(Strategy):
    def __init__(self):
        self.teacher = None
        self.old_classes = []       # labels the teacher knows
        self.kd_loss = 0.0

    def loss(self, model, x, y):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if self.teacher is None:                # first experience: no teacher yet
            return loss
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        T = cfg.KD_TEMPERATURE
        student = F.log_softmax(logits[:, self.old_classes].float() / T, dim=1)
        teacher = F.softmax(teacher_logits[:, self.old_classes].float() / T, dim=1)
        kd = F.kl_div(student, teacher, reduction="batchmean") * T * T
        self.kd_loss = kd.item()
        return loss + cfg.KD_LAMBDA * kd

    def after_experience(self, model, dataset):
        self.old_classes = sorted(set(self.old_classes) | {label for _, label in dataset.items})
        self.teacher = copy.deepcopy(model).eval()          # frozen snapshot = the teacher for the next experience
        for p in self.teacher.parameters():
            p.requires_grad = False

    def status(self):
        return f"(distillation loss {self.kd_loss:.4f})" if self.teacher is not None else ""
