import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # .../continual_learning
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]                          # + project root (models.py, dataset.py, ...)

from cl_common.runner import run_experiment
from naive.strategy import Naive

if __name__ == "__main__":
    run_experiment(Naive(), "naive")
