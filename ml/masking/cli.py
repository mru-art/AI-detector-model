"""Cloak an artwork before posting it, and print its health score.

    python -m ml.masking.cli art.png -o art.cloaked.png --checkpoint runs/best_model.pth
    python -m ml.masking.cli art.png -o out.png --checkpoint best.pth --decoy other.png
"""

from __future__ import annotations

import argparse

from PIL import Image

from .cloak import CloakConfig, cloak, image_ssim
from .encoder import load_scram_encoder
from .health import protection_health


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloak an artwork against AI scrapers.")
    parser.add_argument("image", help="Artwork to protect")
    parser.add_argument("-o", "--output", required=True, help="Where to write the cloaked PNG")
    parser.add_argument("--checkpoint", required=True, help="SCRAM model checkpoint (.pth)")
    parser.add_argument("--decoy", help="Optional image to move the features toward")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Max pixel change, out of 255 (default 8)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--min-ssim", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = CloakConfig(epsilon=args.epsilon / 255.0, steps=args.steps, min_ssim=args.min_ssim)
    encoder = load_scram_encoder(args.checkpoint, device=args.device)

    original = Image.open(args.image).convert("RGB")
    decoy = Image.open(args.decoy).convert("RGB") if args.decoy else None
    protected = cloak(original, encoder, config, decoy=decoy, device=args.device)
    protected.save(args.output, format="PNG")

    report = protection_health(original, protected, encoder, device=args.device)
    worst_name, worst_value = report.worst_attacker if report.per_attacker else ("n/a", 0.0)
    print(f"Saved: {args.output}")
    print(f"SSIM: {image_ssim(original, protected):.4f}")
    print(f"Health: {report.score:.1f}/100")
    print(f"Weakest attack: {worst_name} ({worst_value:.1f})")


if __name__ == "__main__":
    main()
