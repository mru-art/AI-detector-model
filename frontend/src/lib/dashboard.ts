import { USE_MOCK, request } from "@/lib/api";
import snapshot from "@/data/metrics-snapshot.json";

/** Response of GET /metrics/continual (backend/metrics.py). Accuracies are 0–1; null = not learned yet. */
export type Metrics = {
  method: string;
  method_label: string;
  smoke_test: boolean;
  methods: { id: string; label: string; smoke: boolean }[];
  classes: { id: string; label: string }[];
  experiences: { id: string; generator: string; label: string }[];
  acc: (number | null)[][];
  ai_det: (number | null)[][];
  avg_acc: number[];
  avg_ai_det: number[];
  forgetting: (number | null)[];
  summary: Record<string, number | null>;
};

export type MetricsResult = { data: Metrics; live: boolean };

/**
 * Live numbers from the API; if it can't be reached, the snapshot exported from the
 * repo's results (python -m backend.metrics > frontend/src/data/metrics-snapshot.json).
 */
export async function fetchMetrics(method: string | null, signal?: AbortSignal): Promise<MetricsResult> {
  if (!USE_MOCK) {
    try {
      const q = method ? `?method=${encodeURIComponent(method)}` : "";
      return { data: await request<Metrics>(`/metrics/continual${q}`, { signal }), live: true };
    } catch (err) {
      if ((err as Error).name === "AbortError") throw err;
    }
  }
  return { data: snapshot as Metrics, live: false };
}

export type GeneratorSummary = {
  id: string;
  label: string;
  experience: string;
  learned: number;
  now: number;
  change: number;
};

/** For each generator: accuracy right after it was learned vs after the last experience. */
export function perGenerator(m: Metrics, key: "acc" | "ai_det"): GeneratorSummary[] {
  const rows = m[key];
  const last = rows.length - 1;
  return m.experiences.map((e, i) => {
    const col = m.classes.findIndex((c) => c.id === e.generator);
    const learned = rows[i]?.[col] ?? 0;
    const now = rows[last]?.[col] ?? 0;
    return { id: e.generator, label: e.label, experience: e.id, learned, now, change: now - learned };
  });
}
