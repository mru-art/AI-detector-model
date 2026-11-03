// Our own result shape for the UI. The backend contract is not agreed yet,
// so this is where the adapter goes once it is.

/** A flagged region, normalised to 0–1 of the image width/height. */
export type Region = { x: number; y: number; w: number; h: number };

export type AnalysisResult = {
  verdict: "real" | "ai";
  /** 0–1 */
  confidence: number;
  likelySource: string | null;
  regions: Region[];
  /** True while results are generated locally, not by the model. */
  isMock: boolean;
};

export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";

/**
 * Set to true once the trained model replaces the backend's placeholder
 * result in backend/api/detections.py. Until then, real (non-mock) results
 * show a "preview" notice.
 */
export const MODEL_CONNECTED = false;

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  "",
);

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

/**
 * MOCK: waits ~2.5 s, then returns a fake result.
 * Verdict follows file-size parity (even = AI, odd = real) so both
 * result states can be tested. None of this comes from a model.
 */
async function mockAnalyze(file: File, signal?: AbortSignal): Promise<AnalysisResult> {
  await wait(2500, signal);
  const ai = file.size % 2 === 0;
  return {
    verdict: ai ? "ai" : "real",
    confidence: ai ? 0.87 : 0.74,
    likelySource: ai ? "Midjourney" : null,
    regions: ai
      ? [
          { x: 0.12, y: 0.18, w: 0.3, h: 0.25 },
          { x: 0.55, y: 0.5, w: 0.28, h: 0.3 },
        ]
      : [],
    isMock: true,
  };
}

/** Shape of POST /detect (backend/api/schemas.py → DetectionResponse). */
type DetectionResponse = {
  prediction: string;
  ai_probability: number;
  confidence: number;
  generator: string;
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
    if (Array.isArray(body.detail)) {
      return body.detail.map((e: { msg: string }) => e.msg).join("; ");
    }
  } catch {
    /* not JSON */
  }
  return `The detector returned an error (${response.status}).`;
}

async function realAnalyze(file: File, signal?: AbortSignal): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/detect`, { method: "POST", body: form, signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new Error(
      "Couldn’t reach the detector. If it hasn’t been used for a while it may be waking up, so try again in a minute.",
    );
  }

  if (!response.ok) throw new Error(await errorMessage(response));
  return toResult((await response.json()) as DetectionResponse);
}

export async function analyzeImage(file: File, signal?: AbortSignal): Promise<AnalysisResult> {
  if (USE_MOCK) return mockAnalyze(file, signal);
  return realAnalyze(file, signal);
}
