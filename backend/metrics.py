"""Continual-learning results for the dashboard, read from the metrics.json files that
vision_transformer/continual_learning writes (results/<method>/metrics.json).

Export a snapshot for the frontend (used when the API can't be reached):
    python -m backend.metrics > frontend/src/data/metrics-snapshot.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .inference import VIT_DIR, pretty_name

RESULTS_ROOT = VIT_DIR / "continual_learning"
METHOD_LABELS = {
    "replay": "Experience replay",
    "ewc": "EWC",
    "kd": "Knowledge distillation",
    "replay_kd": "Replay + distillation",
    "naive": "Naive fine-tuning",
}


def available_methods() -> list[dict[str, Any]]:
    """Full runs first; smoke tests only for methods without a full run."""
    seen: dict[str, dict[str, Any]] = {}
    for folder, smoke in (("results", False), ("results_smoke", True)):
        for f in sorted((RESULTS_ROOT / folder).glob("*/metrics.json")):
            method = f.parent.name
            if method not in seen:
                seen[method] = {
                    "id": method,
                    "label": METHOD_LABELS.get(method, method.replace("_", " ").title()),
                    "smoke": smoke,
                    "path": f,
                }
    order = list(METHOD_LABELS)
    return sorted(seen.values(), key=lambda m: (m["smoke"], order.index(m["id"]) if m["id"] in order else 99))


def load_metrics(method: str | None = None) -> dict[str, Any] | None:
    methods = available_methods()
    if not methods:
        return None
    chosen = next((m for m in methods if m["id"] == method), None) if method else methods[0]
    if chosen is None:
        raise KeyError(method)

    raw = json.loads(Path(chosen["path"]).read_text())
    generators: list[str] = raw["generators"]
    class_ids = ["real"] + generators

    def matrix(key: str) -> list[list[float | None]]:
        # One row per experience, one column per class; None = not learned yet at that point.
        return [[row.get(c) for c in class_ids] for row in raw.get(key, [])]

    return {
        "method": chosen["id"],
        "method_label": chosen["label"],
        "smoke_test": chosen["smoke"],
        "methods": [{k: v for k, v in m.items() if k != "path"} for m in methods],
        "classes": [{"id": c, "label": pretty_name(c)} for c in class_ids],
        "experiences": [
            {"id": f"E{i + 1}", "generator": g, "label": pretty_name(g)} for i, g in enumerate(generators)
        ],
        "acc": matrix("acc"),
        "ai_det": matrix("ai_det"),
        "avg_acc": raw.get("avg_acc", []),
        "avg_ai_det": raw.get("avg_ai_det", []),
        "forgetting": raw.get("forgetting", []),
        "summary": raw.get("summary", {}),
    }


if __name__ == "__main__":
    print(json.dumps(load_metrics(os.getenv("METHOD")), indent=1))
