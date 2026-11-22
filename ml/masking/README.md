# SCRAM art cloaking

Protect an artwork before you post it. The cloaking module adds a small, hard-to-see pattern to the image. The pattern shifts how SCRAM's ViT encoder represents the image, so a model trained on the picture learns less from it. The approach follows Glaze: the change stays below the visible threshold for people, but the features a model relies on move.

The module also gives a **health score** (0–100). It estimates how much of that feature shift survives common preprocessing a scraper might apply. Higher means the cloak is harder to remove.

> **Read this first.** The cloak is optimized against SCRAM's own encoder. Protection against other companies' models is not measured here, and may be weaker. The health score is an estimate of robustness to simple preprocessing. It is not proof that a scraper can't use the image.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Quick start (command line)](#quick-start-command-line)
- [Choosing settings](#choosing-settings)
- [Reading the health score](#reading-the-health-score)
- [Using it from Python](#using-it-from-python)
- [Running the tests](#running-the-tests)
- [Limitations](#limitations)
- [Files](#files)

## How it works

1. **Encode.** Your SCRAM checkpoint is loaded as a feature encoder. The classification head is removed, so the encoder returns the 768-value feature vector for an image.
2. **Perturb.** The cloak starts from the original image and changes each pixel by at most `epsilon` (default 8 out of 255). It takes 100 signed gradient steps that push the image's features away from its own features. Each step is a small move, and every move is clipped to the budget.
3. **Robust optimization.** By default, each step first resizes the image down and back up with a random scale, and adds a little noise. The perturbation is therefore tuned to survive that kind of change, not only the exact pixels it was built on.
4. **Check fidelity.** The result is rounded to 8-bit colour. If its SSIM against the original is below `min_ssim` (default 0.95), the cloak is rejected with an error.
5. **Score.** The health score runs five simple preprocessing attacks on the cloaked image and reports how much of the feature shift remains after each one.

Optionally, you can pass a **decoy** image. The cloak then moves the image's features toward the decoy's features instead of simply away from its own.

## Requirements

- Python 3.11 or newer
- `torch` and `torchvision`
- `numpy`, `Pillow`, `scikit-image`
- A trained SCRAM checkpoint (`.pth`) in the layout produced by `vision_transformer/models.py`, i.e. the ViT-B/16 with the custom head (`heads.head.0`, `.3`, `.6`).

```bash
pip install torch torchvision numpy pillow scikit-image
```

Checkpoints are git-ignored. Use the one you trained, for example `vision_transformer/runs/<run>/best_model.pth`.

## Quick start (command line)

From the repository root:

```bash
python -m ml.masking.cli artwork.png \
    -o artwork.cloaked.png \
    --checkpoint vision_transformer/runs/<run>/best_model.pth
```

Output:

```
Saved: artwork.cloaked.png
SSIM: 0.9782
Health: 48.3/100
Weakest attack: Reconstruct (12.4)
```

Upload `artwork.cloaked.png`, not the original. Keep the original file; the cloaked copy cannot be turned back into it.

### All options

| Option | Default | What it does |
|---|---|---|
| `image` | (required) | The artwork to protect |
| `-o`, `--output` | (required) | Where to write the cloaked PNG. Use PNG, not JPEG, so no new compression is added |
| `--checkpoint` | (required) | SCRAM model checkpoint (`.pth`) |
| `--decoy` | none | Image whose features the cloak should move toward |
| `--epsilon` | `8` | Maximum change per pixel, out of 255. Larger is stronger and more visible |
| `--steps` | `100` | Optimization steps. More is slower and usually stronger |
| `--min-ssim` | `0.95` | Visual fidelity floor. The run fails if the result is below this |
| `--device` | `cpu` | `cpu` or `cuda` |

## Choosing settings

- **Start with the defaults.** Check the result at full size and on a phone screen.
- **If the image looks changed,** lower `--epsilon` to 4 or 6. Expect a weaker cloak.
- **If the run fails with "below the floor",** the image is too sensitive for that strength. Lower `--epsilon`, or accept a lower `--min-ssim` only after you've looked at the output.
- **Flat, simple images** (plain colour, large empty areas) show a pattern more easily than detailed ones. Test those with a smaller epsilon.
- **On a CPU,** 100 steps on a 1024×1024 image can take a long time. Test on a smaller crop to find settings, then run the full image.

## Reading the health score

The score is the average over five attacks:

| Attack | What it simulates |
|---|---|
| Identity | Downloading the file unchanged |
| Screenshot | Displaying the image in a 1280×720 browser window, re-saved as JPEG at quality 85 |
| Recompress | Light blur, downscale to 75%, JPEG at quality 70 |
| Denoise | A 3×3 median filter, a common noise-removal step |
| Reconstruct | Downscale to 25% and scale back up, similar to an autoencoder bottleneck |

For each attack, the score records what fraction of the cloak's feature shift is still there after the attack, compared with the clean image. The result is a number from 0 to 100.

Rough guide:

- **Under 20:** the cloak is removed by most of these attacks. It's probably not worth posting.
- **20–50:** some protection remains. Try a higher `--epsilon` or more `--steps`.
- **Above 50:** the feature shift survives most simple preprocessing.

The printed **weakest attack** is the one that removes the most shift. If it's Recompress or Screenshot, the image is likely to be resized or re-encoded by a platform. Platforms do that a lot, so this is useful to know.

Identity is listed but **not averaged in**: it always keeps the full shift, so including it would inflate every score. The score is the mean of the other four attacks.

The robust mode (on by default) optimizes through resizing, blur, coarse colour quantization and noise. On a synthetic test image with a stand-in encoder, it raised the score from about 6 to about 55 at the same fidelity. That is a result about the optimization, not about your model; check it on your checkpoint before relying on it.

## Using it through the API

With the backend running (`uvicorn backend.api.main:app`), send an image to `POST /cloak`:

```bash
curl -F "file=@artwork.png" -F "epsilon=8" -F "steps=100" http://localhost:8000/cloak
```

The response is JSON: `image_png_base64` (the cloaked PNG), `ssim`, `health`, `per_attacker`, `weakest_attack`. Nothing is stored on the server. `steps` is capped by `CLOAK_MAX_STEPS` (default 300). The default is `CLOAK_STEPS` (default 100). Only one cloaking job runs at a time, because it is CPU-heavy.

## Using it from Python

```python
from PIL import Image
from ml.masking import CloakConfig, cloak, image_ssim, load_scram_encoder, protection_health

encoder = load_scram_encoder("vision_transformer/runs/<run>/best_model.pth", device="cpu")

artwork = Image.open("artwork.png").convert("RGB")

config = CloakConfig(
    epsilon=8 / 255,   # max change per pixel
    steps=100,         # optimization steps
    min_ssim=0.95,     # fidelity floor; 0 disables the check
    robust=True,       # optimize through resizing and noise
)

cloaked = cloak(artwork, encoder, config, device="cpu")
cloaked.save("artwork.cloaked.png")

print("SSIM:", round(image_ssim(artwork, cloaked), 4))
report = protection_health(artwork, cloaked, encoder)
print("Health:", report.score)
print("Per attack:", report.per_attacker)
print("Weakest:", report.worst_attacker)
```

Pass `decoy=` to `cloak()` to move features toward another image:

```python
decoy = Image.open("other_artwork.png").convert("RGB")
cloaked = cloak(artwork, encoder, config, decoy=decoy)
```

`load_scram_encoder` raises `ValueError` if the checkpoint isn't in the expected layout.

## Running the tests

```bash
python -m pytest tests/test_masking.py
```

The tests use a small stand-in encoder, so they run in a few seconds without your checkpoint. They check the perturbation budget, that the cloak moves features in the right direction, the fidelity floor, and that the health score stays within 0–100.

## Limitations

- **Optimized against SCRAM's encoder only.** Whether the cloak affects other companies' models is not tested here. Treat it as unproven until tested against the models you care about.
- **The health score is an estimate.** It uses five fixed preprocessing steps, not learned scrapers, and it measures SCRAM's own features.
- **The Identity attack is always part of the average** (see the caveat above), so the absolute score is hard to interpret on its own.
- **It doesn't survive everything.** Heavy cropping, strong compression, or re-drawing the image can remove the pattern.
- **Visibility depends on the image and the viewing conditions.** Check the output on the screens and sizes your audience uses.
- **CPU runs are slow.** A GPU speeds up optimization considerably.

## Files

| File | Purpose |
|---|---|
| `cloak.py` | The optimization, `CloakConfig`, the SSIM check |
| `encoder.py` | Loads a SCRAM checkpoint as a feature encoder |
| `health.py` | The health score and its five attacks |
| `cli.py` | Command-line entry point |
| `../../backend/api/cloak.py` | The `POST /cloak` endpoint |
| `__init__.py` | Public imports |
