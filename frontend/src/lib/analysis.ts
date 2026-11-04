import { USE_MOCK, request, wait } from "@/lib/api";

/** A crop of the image, as fractions (0–1) of its width and height. */
export type Region = { x: number; y: number; w: number; h: number; ai_probability: number };

export type Source = { id: string; label: string; probability: number; is_real: boolean };

/** Response of POST /detect (backend/api/schemas.py → DetectionResponse). */
export type AnalysisResult = {
  id: string | null;
  model_version: string;
  prediction: "AI" | "REAL";
  ai_probability: number;
  confidence: number;
  generator: string | null;
  sources: Source[];
  uncertainty: number;
  crops: { agree: number; total: number };
  tiles: { rows: number; cols: number; values: number[][]; regions: Region[] } | null;
  notes: string[];
  image: { width: number; height: number };
  elapsed_ms: number;
  timestamp: string;
  /** Added by the frontend: true when the result was made up locally. */
  isMock?: boolean;
};

export function confidenceWord(p: number): string {
  const c = Math.max(p, 1 - p);
  return c >= 0.95 ? "very high" : c >= 0.85 ? "high" : c >= 0.7 ? "moderate" : "low";
}

/** MOCK (VITE_USE_MOCK=true): a made-up result in the real shape, for UI work without a backend. */
async function mockAnalyze(file: File, signal?: AbortSignal): Promise<AnalysisResult> {
  await wait(1800, signal);
  const ai = file.size % 2 === 0;
  const p = ai ? 0.91 : 0.12;
  const rows = 3;
  const cols = 4;
  const values = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => Math.min(0.99, Math.max(0.01, p + Math.sin(r * 2 + c) * 0.25))),
  );
  const regions = values.flatMap((row, r) =>
    row.map((v, c) => ({ x: c / cols, y: r / rows, w: 1 / cols, h: 1 / rows, ai_probability: v })),
  );
  const sources: Source[] = [
    { id: "real", label: "Real / human-made", probability: 1 - p, is_real: true },
    { id: "midjourney", label: "Midjourney", probability: p * 0.62, is_real: false },
    { id: "sdxl", label: "Stable Diffusion XL", probability: p * 0.23, is_real: false },
    { id: "dalle3", label: "DALL·E 3", probability: p * 0.15, is_real: false },
  ].sort((a, b) => b.probability - a.probability);
  return {
    id: null,
    model_version: "mock",
    prediction: ai ? "AI" : "REAL",
    ai_probability: p,
    confidence: Math.max(p, 1 - p),
    generator: ai ? "Midjourney" : null,
    sources,
    uncertainty: 0.31,
    crops: { agree: 5, total: 5 },
    tiles: { rows, cols, values, regions },
    notes: [],
    image: { width: 1200, height: 900 },
    elapsed_ms: 1800,
    timestamp: new Date().toISOString(),
    isMock: true,
  };
}

export async function analyzeImage(file: File, signal?: AbortSignal): Promise<AnalysisResult> {
  if (USE_MOCK) return mockAnalyze(file, signal);
  const form = new FormData();
  form.append("file", file);
  return request<AnalysisResult>("/detect", { method: "POST", body: form, signal });
}

/** Hand-off for a file dropped on the home page, picked up by the Analyze page. */
let pending: File | null = null;
export function handOff(file: File): void {
  pending = file;
}
export function takeHandOff(): File | null {
  const f = pending;
  // Cleared on the next tick, not now: React StrictMode mounts the page twice in development,
  // and the second mount (after the first one's cleanup cancelled it) must still find the file.
  if (f) window.setTimeout(() => { if (pending === f) pending = null; }, 0);
  return f;
}
