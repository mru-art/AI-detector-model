# Distillation loop: add new generators without retraining

This loop adds a new AI generator to SCRAM's detector using a few stored images of each class you already know, plus a frozen copy of the current model that keeps its old answers unchanged. You don't retrain on the whole dataset each time a new generator appears.

## How it works

When a new generator (say `sd3`) appears, the loop does this:

1. **Freeze the teacher.** The saved model, as it is now, is copied and frozen. It is the reference for the old classes.
2. **Add an output.** The student gets one more output for the new generator. Its new row starts fresh.
3. **Warm up the new output.** A short head-only pass trains just the classifier (`HEAD_EPOCHS`), with the backbone frozen.
4. **Train on new data plus exemplars.** The student trains on the new generator's images and on a small memory of old images, `DISTILL_EXEMPLARS_PER_CLASS` per class (default 200). Cross-entropy covers all labels.
5. **Keep the old answers.** For every image in every batch, the student's scores on the old classes are pulled toward the teacher's (knowledge distillation, weight `KD_LAMBDA`). This is what stops the model forgetting `real`, `dalle3`, and the others.
6. **Store exemplars of the new generator.** A few of its images join the memory, so the next update can keep it too.
7. **Save and evaluate.** The model, memory, and metrics are written after every generator, and the model is evaluated on every class it knows.

Only paths are stored in the memory (`results/distill_loop/exemplars.json`), not pixels, so the memory is tiny. Old classes keep the same exemplars every time, because each one is selected once with a fixed seed.

## Using it

Put the new generator in `cl_common/config.py` (append it to `CLASS_TO_IDX`, `NUM_CLASSES`, and `GENERATORS`; never reorder existing entries), then run from `vision_transformer/continual_learning`:

```
python distill_loop\run.py                 # starts from results/replay_kd/ the first time
python distill_loop\run.py --from ewc      # or start from another method's saved model
```

The first run builds the memory from the existing training images of the classes the model already knows. Later runs continue from `results/distill_loop/`, so you only need to run it again after you append to `GENERATORS`. If you run it when nothing is pending, it stops with a message.

Output goes to `results/distill_loop/`: `final_model.pth` (the previous one is kept as `previous_model.pth`), `exemplars.json`, and the usual `metrics.json`, `tables.txt` and plots.

## Settings (in `cl_common/config.py`)

| Setting | Default | Effect |
|---|---|---|
| `DISTILL_EXEMPLARS_PER_CLASS` | 200 | Old images kept per class. More means better retention, slower epochs |
| `DISTILL_EPOCHS` | 3 | Epochs on the new generator's images plus exemplars |
| `KD_LAMBDA` | 2.0 | Strength of keeping the old answers. Raise it if old classes drop; lower it if the new generator is learned poorly |
| `KD_TEMPERATURE` | 2.0 | Softness of the teacher's answers |
| `HEAD_EPOCHS`, `LR_HEAD` | 1, 3e-4 | Head-only warm-up before the full update |
| `LR`, `LAYER_DECAY` | 5e-5, 0.8 | Learning rate for the backbone; lower layers change less |

## Checking it works

- Compare `results/distill_loop/tables.txt` with `results/replay_kd/tables.txt`: the old classes' accuracy should not fall much after each update, and the new generator should be learned.
- `previous_model.pth` is the model before the last update, so you can roll back one step.
- Tests: `python -m pytest distill_loop\test_loop.py`. They use random weights and a tiny synthetic dataset, so they check the mechanics (memory, resuming, the KD term being used), not accuracy.

## Limits

- **Accuracy is not measured yet.** The tests run without your dataset. Whether a few hundred exemplars per class keep old accuracy is what the real comparison against `replay_kd` (which keeps 5000 per generator) needs to show.
- **Exemplars are chosen at random.** Herding (choosing images that best represent each class's features) often retains more, but isn't implemented.
- **Exemplars are the only defence against drift in the real class.** If the real images are very different from the generators, use a larger exemplar count.
- **The memory points at files on disk.** If the dataset moves, the loop stops and asks you to delete `exemplars.json` so it can rebuild.
