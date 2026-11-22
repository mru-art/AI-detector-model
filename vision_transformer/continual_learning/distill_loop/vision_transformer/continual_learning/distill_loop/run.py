"""Add new generators to the model with knowledge distillation and a few stored old images.

    python distill_loop\\run.py                  # continue from the replay_kd model (or from distill_loop once it exists)
    python distill_loop\\run.py --from ewc       # start from another method's saved model

Every generator in config.py that the model has not learned yet is added, one at a time.
Run it again after you append a new generator to config.py."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]

from cl_common.runner import CLError                  # noqa: E402
from distill_loop.loop import NAME, run_update        # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", default="replay_kd",
                        help="results folder of the model to start from, if distill_loop does not exist yet")
    args = parser.parse_args()
    try:
        run_update(args.source, NAME)
    except CLError as e:
        raise SystemExit(f"\n{e}")
