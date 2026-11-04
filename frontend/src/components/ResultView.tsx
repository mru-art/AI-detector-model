import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Download, Info } from "lucide-react";

import { confidenceWord, type AnalysisResult, type Region } from "@/lib/analysis";
import { pct } from "@/lib/files";

type Overlay = "map" | "boxes" | "off";

/** Re-render when the site theme changes (colours come from CSS variables). */
function useIsDark(): boolean {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(document.documentElement.classList.contains("dark")));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return dark;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.trim().replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Diverging: human colour -> transparent neutral at 0.5 -> AI colour. */
function tileColor(p: number, human: number[], ai: number[]): string {
  const side = p >= 0.5 ? ai : human;
  const strength = Math.min(1, Math.abs(p - 0.5) * 2);
  return `rgba(${side[0]},${side[1]},${side[2]},${(0.12 + strength * 0.5).toFixed(3)})`;
}

/** The uploaded artwork on its mat, with the region map / boxes and a hover readout. */
export function ArtworkStage({
  url,
  alt,
  result,
  overlay = "off",
  scanning = false,
}: {
  url: string;
  alt: string;
  result?: AnalysisResult;
  overlay?: Overlay;
  scanning?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dark = useIsDark();
  const [hover, setHover] = useState<Region | null>(null);
  const tiles = result?.tiles ?? null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !tiles || overlay !== "map") return;
    const css = getComputedStyle(document.documentElement);
    const human = hexToRgb(css.getPropertyValue("--human"));
    const ai = hexToRgb(css.getPropertyValue("--ai"));
    canvas.width = tiles.cols;
    canvas.height = tiles.rows;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, tiles.cols, tiles.rows);
    tiles.values.forEach((row, r) =>
      row.forEach((v, c) => {
        ctx.fillStyle = tileColor(v, human, ai);
        ctx.fillRect(c, r, 1, 1);
      }),
    );
  }, [tiles, overlay, dark]);

  function onMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!tiles || overlay === "off") return;
    const box = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - box.left) / box.width;
    const y = (e.clientY - box.top) / box.height;
    // Nearest tile centre: tiles overlap on small images, so "contains" would be ambiguous.
    let best: Region | null = null;
    let bestD = Infinity;
    for (const r of tiles.regions) {
      const d = (r.x + r.w / 2 - x) ** 2 + (r.y + r.h / 2 - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = r;
      }
    }
    setHover(best);
  }

  const flagged = tiles?.regions.filter((r) => r.ai_probability >= 0.5) ?? [];

  return (
    <div className="flex items-center justify-center rounded-2xl bg-sunken p-4 sm:p-8">
      <div
        className="relative w-fit max-w-full overflow-hidden shadow-[0_1px_2px_rgb(0_0_0/0.12),0_12px_32px_-12px_rgb(0_0_0/0.35)]"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        <img src={url} alt={alt} className="block max-h-[68svh] w-auto max-w-full" />

        {tiles && overlay === "map" && (
          <canvas
            ref={canvasRef}
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 size-full"
            style={{ imageRendering: "auto", filter: "blur(6px)", transform: "scale(1.04)" }}
          />
        )}

        {tiles && overlay === "boxes" && (
          <div className="pointer-events-none absolute inset-0" aria-hidden="true">
            {flagged.map((r, i) => (
              <div
                key={i}
                className="absolute border-2 border-ai outline outline-1 outline-white/70"
                style={{ left: `${r.x * 100}%`, top: `${r.y * 100}%`, width: `${r.w * 100}%`, height: `${r.h * 100}%` }}
              />
            ))}
          </div>
        )}

        {hover && overlay !== "off" && (
          <>
            <div
              className="pointer-events-none absolute border-2 border-white mix-blend-difference"
              style={{
                left: `${hover.x * 100}%`,
                top: `${hover.y * 100}%`,
                width: `${hover.w * 100}%`,
                height: `${hover.h * 100}%`,
              }}
              aria-hidden="true"
            />
            <div
              className="pointer-events-none absolute rounded-lg bg-ink px-2.5 py-1.5 text-sm text-paper shadow-lg"
              style={{
                left: `${Math.min(hover.x + hover.w / 2, 0.8) * 100}%`,
                top: `${Math.min(hover.y + hover.h, 0.9) * 100}%`,
                transform: "translate(-50%, 6px)",
              }}
            >
              <span className="tabular font-semibold">{pct(hover.ai_probability)}</span> AI in this area
            </div>
          </>
        )}

        {scanning && (
          <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
            <div className="absolute inset-0 bg-paper/25" />
            <div
              className="absolute inset-x-0 h-full bg-gradient-to-b from-transparent via-human/35 to-transparent"
              style={{ animation: "scram-scan 1.6s cubic-bezier(.45,0,.55,1) infinite alternate" }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

/** One scale from human-made to AI-generated with the verdict marked on it. */
function ProvenanceScale({ p }: { p: number }) {
  return (
    <div>
      <div className="relative h-3 rounded-full" style={{
        background: "linear-gradient(90deg, var(--human), color-mix(in oklab, var(--human) 25%, var(--line)) 42%, var(--line) 50%, color-mix(in oklab, var(--ai) 25%, var(--line)) 58%, var(--ai))",
      }}>
        <div
          className="absolute top-1/2 size-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-surface bg-ink shadow"
          style={{ left: `${p * 100}%` }}
          role="img"
          aria-label={`${pct(p)} likely AI-generated`}
        />
      </div>
      <div className="mt-2 flex justify-between text-sm text-muted">
        <span>Human-made</span>
        <span>AI-generated</span>
      </div>
    </div>
  );
}

function SourceBreakdown({ result }: { result: AnalysisResult }) {
  const [all, setAll] = useState(false);
  const shown = all ? result.sources : result.sources.slice(0, 5);
  return (
    <div>
      <h3 className="text-base font-bold">Most likely sources</h3>
      <ul className="mt-3 space-y-2.5">
        {shown.map((s) => (
          <li key={s.id}>
            <div className="flex items-baseline justify-between gap-3 text-[15px]">
              <span className="truncate">{s.label}</span>
              <span className="tabular shrink-0 font-semibold">{pct(s.probability, s.probability < 0.1 ? 1 : 0)}</span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-sunken">
              <div
                className={`h-full rounded-full ${s.is_real ? "bg-human" : "bg-ai"}`}
                style={{ width: `${Math.max(1, s.probability * 100)}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
      {result.sources.length > 5 && (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className="mt-3 cursor-pointer text-sm font-semibold text-human hover:underline"
        >
          {all ? "Show fewer" : `Show all ${result.sources.length}`}
        </button>
      )}
    </div>
  );
}

function downloadReport(result: AnalysisResult, fileName: string) {
  const blob = new Blob([JSON.stringify({ file: fileName, ...result }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${fileName.replace(/\.[^.]+$/, "")}-scram-report.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

type Props = { url: string; fileName: string; result: AnalysisResult; onReset: () => void };

export default function ResultView({ url, fileName, result, onReset }: Props) {
  const isAi = result.prediction === "AI";
  const hasTiles = !!result.tiles && result.tiles.regions.length > 1;
  const [overlay, setOverlay] = useState<Overlay>(hasTiles ? "map" : "off");
  const flaggedCount = useMemo(
    () => result.tiles?.regions.filter((r) => r.ai_probability >= 0.5).length ?? 0,
    [result],
  );

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1.55fr)_minmax(320px,1fr)]">
      <div>
        <ArtworkStage url={url} alt={`Analyzed image: ${fileName}`} result={result} overlay={overlay} />
        {hasTiles && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <div className="inline-flex rounded-full bg-sunken p-1" role="group" aria-label="Image overlay">
              {(
                [
                  ["map", "Region map"],
                  ["boxes", `AI-looking areas (${flaggedCount})`],
                  ["off", "Plain image"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  aria-pressed={overlay === id}
                  onClick={() => setOverlay(id)}
                  className={`cursor-pointer rounded-full px-3.5 py-1.5 text-sm transition-colors ${
                    overlay === id ? "bg-surface font-semibold shadow-sm" : "text-muted hover:text-ink"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            {overlay === "map" && (
              <p className="flex items-center gap-2 text-sm text-muted">
                <span className="size-3 rounded-sm bg-human/60" aria-hidden="true" /> human-looking
                <span className="ml-2 size-3 rounded-sm bg-ai/60" aria-hidden="true" /> AI-looking
              </p>
            )}
          </div>
        )}
        {hasTiles && (
          <p className="mt-3 max-w-xl text-sm text-muted">
            The model checked {result.tiles!.regions.length} patches of the original pixels. Hover over the image to
            see the score for each area.
          </p>
        )}
      </div>

      <div className="flex flex-col gap-6" aria-live="polite">
        {result.isMock && (
          <div className="flex items-start gap-3 rounded-xl border border-warn/40 bg-warn/10 p-4 text-sm">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden="true" />
            <p>
              <strong>Demo result.</strong> The site is in demo mode, so this was made up and is not a prediction.
            </p>
          </div>
        )}

        <section className="rounded-2xl border border-line bg-surface p-6">
          <div className="flex items-center gap-3">
            <span className={`h-9 w-1.5 rounded-full ${isAi ? "bg-ai" : "bg-human"}`} aria-hidden="true" />
            <h2 className="text-3xl font-extrabold leading-tight tracking-tight sm:text-4xl">
              {isAi ? "Likely AI-generated" : "Likely human-made"}
            </h2>
          </div>
          <p className="mt-2 text-muted">
            <span className="tabular font-semibold text-ink">{pct(result.ai_probability)}</span> chance it was made by
            an AI model. Confidence is {confidenceWord(result.ai_probability)}.
          </p>
          <div className="mt-6">
            <ProvenanceScale p={result.ai_probability} />
          </div>

          <dl className="mt-6 grid grid-cols-[1fr_auto] gap-x-6 gap-y-2.5 border-t border-line pt-5 text-[15px]">
            {result.generator && (
              <>
                <dt className="text-muted">Closest generator</dt>
                <dd className="text-right font-semibold">{result.generator}</dd>
              </>
            )}
            <dt className="text-muted">Crops that agree</dt>
            <dd className="tabular text-right font-semibold">
              {result.crops.agree} of {result.crops.total}
            </dd>
            <dt className="text-muted">Image size</dt>
            <dd className="tabular text-right font-semibold">
              {result.image.width} × {result.image.height}
            </dd>
            <dt className="text-muted">Time taken</dt>
            <dd className="tabular text-right font-semibold">{(result.elapsed_ms / 1000).toFixed(1)} s</dd>
          </dl>
        </section>

        {result.notes.length > 0 && (
          <ul className="space-y-2">
            {result.notes.map((n) => (
              <li key={n} className="flex gap-3 rounded-xl bg-sunken p-3.5 text-[15px]">
                <Info className="mt-0.5 size-4 shrink-0 text-muted" aria-hidden="true" />
                {n}
              </li>
            ))}
          </ul>
        )}

        <section className="rounded-2xl border border-line bg-surface p-6">
          <SourceBreakdown result={result} />
        </section>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={onReset}
            className="inline-flex cursor-pointer items-center justify-center rounded-full bg-ink px-4 py-2.5 text-[15px] font-semibold text-paper transition-opacity hover:opacity-85"
          >
            Analyze another image
          </button>
          <button
            type="button"
            onClick={() => downloadReport(result, fileName)}
            className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-full border border-line px-4 py-2.5 text-[15px] font-semibold transition-colors hover:bg-sunken"
          >
            <Download className="size-4" aria-hidden="true" />
            Download report
          </button>
        </div>

        <p className="text-sm leading-relaxed text-muted">
          This is an estimate, not proof. Resizing, screenshots and heavy compression weaken the traces the detector
          looks for, and it is less reliable on generators it hasn’t learned. Model: {result.model_version}.
        </p>
      </div>
    </div>
  );
}
