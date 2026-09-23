"""Stratified replay buffer for continual learning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ReplaySample:
    """One labeled example retained for continual-learning replay."""

    image_id: str
    image_uri: str
    label: int  # 0 = Real, 1 = AI
    generator: str
    model_version: str
    embedding: NDArray[np.floating]
    confidence: float
    loss: float
    creation_timestamp: float


class StratifiedReplayBuffer:
    """Capacity-limited replay storage with class and generator quotas.

    REAL samples receive 25% of capacity. The remaining 75% is split as
    evenly as possible among registered AI generators. Any remainder slots
    are assigned in the order generators were registered.

    When a quota is full, a sample competes with a weighted choice of its
    current residents. The retention score favors high loss and confidence
    values near 0.5.

    Args:
        max_capacity: Total maximum number of stored samples.
        generators: Registered AI generator labels.
        seed: Optional NumPy random seed for reproducible selection.
    """

    def __init__(
        self,
        max_capacity: int = 20_000,
        generators: Iterable[str] = (
            "SDXL",
            "FLUX",
            "DALL-E",
            "Midjourney",
            "Unknown",
        ),
        seed: int | None = None,
    ) -> None:
        if max_capacity <= 0:
            raise ValueError("max_capacity must be positive")

        self.max_capacity = max_capacity
        self.generators = tuple(dict.fromkeys(generators))
        if not self.generators:
            raise ValueError("At least one AI generator must be registered")

        real_quota = int(max_capacity * 0.25)
        ai_capacity = max_capacity - real_quota
        base, remainder = divmod(ai_capacity, len(self.generators))

        self.quotas: dict[str, int] = {"REAL": real_quota}
        for index, generator in enumerate(self.generators):
            self.quotas[f"AI:{generator}"] = base + (index < remainder)

        self._buckets: dict[str, list[ReplaySample]] = {key: [] for key in self.quotas}
        self._rng = np.random.default_rng(seed)
        self._embedding_dim: int | None = None

    def __len__(self) -> int:
        """Return the number of currently stored samples."""
        return sum(map(len, self._buckets.values()))

    def _bucket_key(self, sample: ReplaySample) -> str:
        if sample.label == 0:
            return "REAL"
        if sample.label == 1:
            key = f"AI:{sample.generator}"
            if key not in self._buckets:
                raise ValueError(
                    f"Unregistered AI generator {sample.generator!r}; "
                    "register it when constructing the buffer"
                )
            return key
        raise ValueError(f"label must be 0 (Real) or 1 (AI), got {sample.label}")

    @staticmethod
    def _retention_scores(samples: list[ReplaySample]) -> NDArray[np.float64]:
        """Combine loss and uncertainty into comparable retention scores."""
        losses = np.asarray([max(0.0, s.loss) for s in samples], dtype=np.float64)
        uncertainty = np.asarray(
            [1.0 - min(1.0, abs(s.confidence - 0.5) * 2.0) for s in samples],
            dtype=np.float64,
        )

        # Normalize loss within the candidate set, then combine with uncertainty.
        loss_range = float(np.ptp(losses))
        normalized_loss = (
            (losses - losses.min()) / loss_range
            if loss_range > 1e-12
            else np.zeros_like(losses)
        )
        return 0.6 * normalized_loss + 0.4 * uncertainty

    def add_sample(self, sample: ReplaySample) -> bool:
        """Add a sample, replacing a lower-value resident if its quota is full.

        Returns:
            True if the sample was added or replaced a resident, False if it
            was rejected because the quota was full and the resident was
            preferable.
        """
        if not 0.0 <= sample.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not math.isfinite(sample.loss):
            raise ValueError("loss must be finite")

        embedding = np.asarray(sample.embedding)
        if embedding.ndim != 1 or embedding.size == 0:
            raise ValueError("embedding must be a non-empty one-dimensional array")
        if not np.issubdtype(embedding.dtype, np.number):
            raise TypeError("embedding must contain numeric values")
        if not np.isfinite(embedding).all():
            raise ValueError("embedding must contain only finite values")

        if self._embedding_dim is None:
            self._embedding_dim = int(embedding.size)
        elif embedding.size != self._embedding_dim:
            raise ValueError(
                f"embedding dimension {embedding.size} does not match "
                f"buffer dimension {self._embedding_dim}"
            )

        key = self._bucket_key(sample)
        bucket = self._buckets[key]
        quota = self.quotas[key]
        if quota == 0:
            return False

        # Store a contiguous numeric array to make later vectorized distance
        # calculations predictable.
        sample.embedding = np.asarray(embedding, dtype=np.float32)

        if len(bucket) < quota:
            bucket.append(sample)
            return True

        candidates = bucket + [sample]
        scores = self._retention_scores(candidates)
        resident_scores = scores[:-1]
        new_score = float(scores[-1])

        # Low-score residents are more likely eviction candidates.
        eviction_weights = np.exp(-4.0 * resident_scores)
        eviction_weights /= eviction_weights.sum()
        victim_index = int(self._rng.choice(len(bucket), p=eviction_weights))

        # Only replace when the new sample is competitive with the selected
        # resident; the stochastic margin avoids deterministic tie churn.
        victim_score = float(resident_scores[victim_index])
        if new_score + self._rng.uniform(0.0, 0.05) < victim_score:
            return False

        bucket[victim_index] = sample
        return True

    def _all_samples(self) -> list[ReplaySample]:
        return [sample for bucket in self._buckets.values() for sample in bucket]

    def _sample_unique(
        self,
        pool: list[ReplaySample],
        count: int,
        selected_ids: set[str],
    ) -> list[ReplaySample]:
        available = [s for s in pool if s.image_id not in selected_ids]
        take = min(count, len(available))
        if take <= 0:
            return []

        indices = self._rng.choice(len(available), size=take, replace=False)
        chosen = [available[int(i)] for i in np.atleast_1d(indices)]
        selected_ids.update(sample.image_id for sample in chosen)
        return chosen

    def _boundary_ranking(self, samples: list[ReplaySample]) -> list[ReplaySample]:
        """Rank samples by cosine distance from their stratum centroid."""
        if not samples:
            return []

        embeddings = np.stack([s.embedding for s in samples]).astype(
            np.float32, copy=False
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-12)

        # Compute one cosine centroid per REAL/AI-generator stratum.
        keys = np.asarray([self._bucket_key(s) for s in samples])
        distances = np.zeros(len(samples), dtype=np.float32)
        for key in np.unique(keys):
            indices = np.flatnonzero(keys == key)
            centroid = normalized[indices].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            distances[indices] = 1.0 - normalized[indices] @ centroid

        ranking = np.argsort(distances)[::-1]
        return [samples[int(i)] for i in ranking]

    def select_replay_batch(self, batch_size: int) -> list[ReplaySample]:
        """Select a unique batch using representative, hard, uncertain,
        and boundary-example strategies.

        The target proportions are 40%, 25%, 20%, and 15%, respectively.
        If the buffer contains fewer samples than requested, all available
        samples are returned in randomized order.
        """
        if batch_size < 0:
            raise ValueError("batch_size cannot be negative")
        samples = self._all_samples()
        if batch_size == 0 or not samples:
            return []

        target_size = min(batch_size, len(samples))
        if target_size < batch_size:
            indices = self._rng.permutation(len(samples))[:target_size]
            return [samples[int(i)] for i in indices]

        portions = np.asarray([0.40, 0.25, 0.20, 0.15]) * target_size
        counts = np.floor(portions).astype(int)
        # Assign rounding remainder to the largest fractional portions.
        remainder = target_size - int(counts.sum())
        for index in np.argsort(portions - counts)[::-1][:remainder]:
            counts[index] += 1

        selected: list[ReplaySample] = []
        selected_ids: set[str] = set()

        # 40% representative: uniformly sample each non-empty stratum,
        # then draw uniformly from the combined stratified pool.
        representative_pool: list[ReplaySample] = []
        nonempty = [bucket for bucket in self._buckets.values() if bucket]
        if nonempty:
            per_bucket = max(1, math.ceil(counts[0] / len(nonempty)))
            for bucket in nonempty:
                representative_pool.extend(
                    self._sample_unique(bucket, per_bucket, set())
                )
        selected.extend(
            self._sample_unique(representative_pool, int(counts[0]), selected_ids)
        )

        # 25% hard examples: largest stored historical losses.
        hard_ranked = sorted(samples, key=lambda s: s.loss, reverse=True)
        selected.extend(self._sample_unique(hard_ranked, int(counts[1]), selected_ids))

        # 20% uncertain examples: confidence closest to 0.5.
        uncertain_ranked = sorted(samples, key=lambda s: abs(s.confidence - 0.5))
        selected.extend(
            self._sample_unique(uncertain_ranked, int(counts[2]), selected_ids)
        )

        # 15% rare/boundary examples: farthest from their stratum centroid.
        boundary_ranked = self._boundary_ranking(samples)
        selected.extend(
            self._sample_unique(boundary_ranked, int(counts[3]), selected_ids)
        )

        # Strategies can overlap. Fill remaining slots uniformly from all
        # unselected samples to preserve the requested batch size.
        if len(selected) < target_size:
            selected.extend(
                self._sample_unique(
                    samples,
                    target_size - len(selected),
                    selected_ids,
                )
            )

        self._rng.shuffle(selected)
        return selected
