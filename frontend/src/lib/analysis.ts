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

export async function analyzeImage(file: File, signal?: AbortSignal): Promise<AnalysisResult> {
  if (USE_MOCK) return mockAnalyze(file, signal);
  throw new Error("The detector API isn’t connected yet.");
}
