// Local drop-in replacement for frontend/src/lib/analysis.ts.
// vite.config.local.ts aliases "@/lib/analysis" to this file, so the repo's
// copy is never edited. Same exports, but analyzeImage calls the backend.

export type Region = { x: number; y: number; w: number; h: number };

export type AnalysisResult = {
  verdict: "real" | "ai";
  /** 0–1 */
  confidence: number;
  likelySource: string | null;
  regions: Region[];
  isMock: boolean;
};

/**
 * Only read by lib/dashboard.ts. The backend has no endpoint for the
 * accuracy matrix that page draws, so the dashboard stays on its mock data.
 */
export const USE_MOCK = true;

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

/** Shape of POST /detect (backend/api/schemas.py → DetectionResponse). */
type DetectionResponse = {
  id: string;
  artwork_id: string;
  model_version: string;
  prediction: "AI" | "REAL" | string;
  ai_probability: number;
  confidence: number;
  generator: string;
  timestamp: string;
};

function toResult(d: DetectionResponse): AnalysisResult {
  const ai = d.prediction.toUpperCase() === "AI";
  return {
    verdict: ai ? "ai" : "real",
    // How sure the model is of the verdict it gave.
    confidence: ai ? d.ai_probability : 1 - d.ai_probability,
    likelySource: ai ? d.generator : null,
    regions: [], // the backend doesn't return regions yet
    isMock: false,
  };
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((e: { msg: string }) => e.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return `The detector returned an error (${response.status}).`;
}

export async function analyzeImage(file: File, signal?: AbortSignal): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/detect`, { method: "POST", body: form, signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new Error("Couldn’t reach the detector API. Is the backend running on port 8000?");
  }

  if (!response.ok) throw new Error(await errorMessage(response));
  return toResult((await response.json()) as DetectionResponse);
}
